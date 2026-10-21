"""
GabayMed face check service.

Compares a selfie with the photo on an uploaded ID and returns how close the
two faces are. It is only a HINT for the Front Desk: it never approves or
rejects anyone, and it never stores or logs the photos (everything is processed
in memory and thrown away when the request ends).

Endpoints
  GET  /health   -> {"ok": true}                     (no key needed)
  POST /compare  -> multipart form, header X-Api-Key
                    fields: selfie (image), id_image (image)
  POST /liveness -> multipart form, header X-Api-Key
                    fields: frames (several images, in order), steps (JSON)
                    Checks that a real person did what was asked: a head turn
                    and a blink, and that it is the same face in every frame.

Lower "distance" = faces are closer. Defaults: <= 0.50 match, <= 0.62 unclear.
Change them with the MATCH_MAX / UNCLEAR_MAX settings after testing with real
sign-ups.
"""
import io
import json
import math
import os
import secrets
import statistics
import threading
from typing import List

import face_recognition
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

API_KEY = os.environ.get("API_KEY", "")
MATCH_MAX = float(os.environ.get("MATCH_MAX", "0.50"))
UNCLEAR_MAX = float(os.environ.get("UNCLEAR_MAX", "0.62"))
MAX_BYTES = int(os.environ.get("MAX_BYTES", str(6 * 1024 * 1024)))
MAX_SIDE = int(os.environ.get("MAX_SIDE", "800"))   # keeps memory under 512 MB (free hosts)

# Liveness (active check) settings. These are starting values: tune them with
# real people on real phones (see README).
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "24"))
FRAME_MAX_BYTES = int(os.environ.get("FRAME_MAX_BYTES", str(600 * 1024)))
LIVE_SIDE = int(os.environ.get("LIVE_SIDE", "480"))
TURN_MIN_SHIFT = float(os.environ.get("TURN_MIN_SHIFT", "0.10"))   # nose shift, as a share of face width
BLINK_MIN_DROP = float(os.environ.get("BLINK_MIN_DROP", "0.07"))   # eye openness must fall by this much
BLINK_MAX_RATIO = float(os.environ.get("BLINK_MAX_RATIO", "0.75")) # ...to at most this share of the open value
SAME_PERSON_MAX = float(os.environ.get("SAME_PERSON_MAX", "0.62"))
# If the turn direction turns out to be reversed on real phones, set this to 1.
TURN_FLIP = os.environ.get("TURN_FLIP", "0") == "1"

app = FastAPI(title="GabayMed face check", docs_url=None, redoc_url=None, openapi_url=None)

# The free host has 512 MB of memory: handle one comparison at a time.
_slots = threading.BoundedSemaphore(1)


def _load(data: bytes, side: int = 0) -> np.ndarray:
    """Decode an upload to an RGB array, honouring phone rotation, size-capped."""
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="unreadable_image")
    cap = side or MAX_SIDE
    img.thumbnail((cap, cap))
    return np.array(img)


def _largest_face(arr: np.ndarray, rotations):
    """Return (encoding or None, number_of_faces_seen) using the biggest face."""
    for turns in rotations:                      # ID cards are often photographed sideways
        rotated = np.ascontiguousarray(np.rot90(arr, k=turns))
        # Second, 4x-bigger pass finds small ID-card photos, but only on small
        # images, otherwise it would use far too much memory.
        passes = (1, 2) if max(rotated.shape[:2]) <= 700 else (1,)
        for upsample in passes:
            locs = face_recognition.face_locations(rotated, number_of_times_to_upsample=upsample)
            if locs:
                biggest = max(locs, key=lambda l: (l[2] - l[0]) * (l[1] - l[3]))
                enc = face_recognition.face_encodings(rotated, [biggest])
                if enc:
                    return enc[0], len(locs)
    return None, 0


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/compare")
def compare(
    selfie: UploadFile = File(...),
    id_image: UploadFile = File(...),
    x_api_key: str = Header(default=""),
):
    if not API_KEY:
        raise HTTPException(status_code=503, detail="not_configured")
    if not secrets.compare_digest(x_api_key.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="bad_key")

    selfie_bytes = selfie.file.read(MAX_BYTES + 1)
    id_bytes = id_image.file.read(MAX_BYTES + 1)
    if len(selfie_bytes) > MAX_BYTES or len(id_bytes) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="too_large")

    selfie_arr = _load(selfie_bytes)
    id_arr = _load(id_bytes)

    if not _slots.acquire(timeout=60):
        raise HTTPException(status_code=503, detail="busy")
    try:
        selfie_enc, selfie_faces = _largest_face(selfie_arr, rotations=(0,))
        if selfie_enc is None:
            return {"status": "no_face_selfie", "distance": None, "selfie_faces": 0, "id_faces": 0}
        id_enc, id_faces = _largest_face(id_arr, rotations=(0, 1, 3, 2))
        if id_enc is None:
            return {"status": "no_face_id", "distance": None, "selfie_faces": selfie_faces, "id_faces": 0}

        distance = float(face_recognition.face_distance([id_enc], selfie_enc)[0])
        if distance <= MATCH_MAX:
            status = "match"
        elif distance <= UNCLEAR_MAX:
            status = "unclear"
        else:
            status = "no_match"
        return {"status": status, "distance": round(distance, 3),
                "selfie_faces": selfie_faces, "id_faces": id_faces}
    finally:
        _slots.release()


# --------------------------------------------------------------------------
# Liveness: did a real person turn their head and blink, as asked?
# --------------------------------------------------------------------------

def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _ear(eye) -> float:
    """Eye openness from its 6 outline points (smaller = more closed)."""
    return (_dist(eye[1], eye[5]) + _dist(eye[2], eye[4])) / (2.0 * max(1e-6, _dist(eye[0], eye[3])))


def _measure(arr: np.ndarray):
    """Largest face in a frame -> {ear, ratio, area, loc}, or None if no face.

    ratio = where the nose sits between the two sides of the jaw (0.5 = facing
    the camera). When the head turns, the nose slides toward one side.
    """
    locs = face_recognition.face_locations(arr, number_of_times_to_upsample=1)
    if not locs:
        return None
    loc = max(locs, key=lambda l: (l[2] - l[0]) * (l[1] - l[3]))
    marks = face_recognition.face_landmarks(arr, [loc], model="large")
    if not marks:
        return None
    lm = marks[0]
    xs = [p[0] for p in lm["chin"]]
    left, right = min(xs), max(xs)
    nose_x = sum(p[0] for p in lm["nose_tip"]) / len(lm["nose_tip"])
    return {
        "ear": (_ear(lm["left_eye"]) + _ear(lm["right_eye"])) / 2.0,
        "ratio": (nose_x - left) / max(1.0, right - left),
        "area": (loc[2] - loc[0]) * (loc[1] - loc[3]),
        "loc": loc,
    }


def _decide(steps, measures, encode):
    """Judge the whole attempt.

    steps    : [{"action": ..., "count": n}, ...]  (frames are in the same order)
    measures : one _measure() result (or None) per frame
    encode   : encode(frame_index) -> 128-number face fingerprint, or None
    """
    checks = {"face_every_step": True, "turn": None, "blink": None, "same_person": True}
    reasons = []

    # Split the frames into their steps.
    segs, pos = [], 0
    for st in steps:
        idx = list(range(pos, pos + st["count"]))
        segs.append((st["action"], idx))
        pos += st["count"]

    # 1. A face must be visible: most of the "straight" frames, and at least one frame per step.
    for action, idx in segs:
        seen = [i for i in idx if measures[i]]
        need = max(1, int(0.6 * len(idx))) if action == "straight" else 1
        if len(seen) < need:
            checks["face_every_step"] = False
    if not checks["face_every_step"]:
        reasons.append("no_face")

    straight = [i for a, idx in segs if a == "straight" for i in idx if measures[i]]
    if not straight:
        return {"passed": False, "checks": checks, "best_frame": None, "reasons": reasons or ["no_face"]}
    base_ratio = statistics.median(measures[i]["ratio"] for i in straight)
    base_ear = statistics.median(measures[i]["ear"] for i in straight)

    # 2. The head turn (only if one was asked for).
    for action, idx in segs:
        if action in ("turn_left", "turn_right"):
            sign = 1 if action == "turn_right" else -1      # turning right moves the nose to the image's left
            if TURN_FLIP:
                sign = -sign
            shifts = [sign * (base_ratio - measures[i]["ratio"]) for i in idx if measures[i]]
            checks["turn"] = bool(shifts) and max(shifts) >= TURN_MIN_SHIFT
            if not checks["turn"]:
                reasons.append("turn_not_seen")

    # 3. The blink (only if one was asked for): the eyes must close, then open again.
    for action, idx in segs:
        if action == "blink":
            ears = [measures[i]["ear"] for i in idx if measures[i]]
            lowest = min(ears) if ears else None
            checks["blink"] = (
                lowest is not None
                and base_ear - lowest >= BLINK_MIN_DROP
                and lowest <= BLINK_MAX_RATIO * base_ear
            )
            if not checks["blink"]:
                reasons.append("blink_not_seen")

    # 4. Same person in every step: compare one frame per step with the first straight frame.
    reference = None
    for action, idx in segs:
        seen = [i for i in idx if measures[i]]
        if not seen:
            continue
        pick = max(seen, key=lambda i: measures[i]["area"])
        enc = encode(pick)
        if enc is None:
            continue
        if reference is None:
            reference = enc
        elif float(np.linalg.norm(reference - enc)) > SAME_PERSON_MAX:
            checks["same_person"] = False
    if not checks["same_person"]:
        reasons.append("different_person")

    # Best front-facing, eyes-open frame, kept as the selfie.
    best = max(
        straight,
        key=lambda i: measures[i]["area"] * (1.0 - min(1.0, abs(measures[i]["ratio"] - 0.5) * 3))
        * (1.0 if measures[i]["ear"] >= 0.9 * base_ear else 0.5),
    )
    ok = (checks["face_every_step"] and checks["same_person"]
          and checks["turn"] is not False and checks["blink"] is not False)
    return {"passed": bool(ok), "checks": checks, "best_frame": best, "reasons": reasons}


@app.post("/liveness")
def liveness(
    frames: List[UploadFile] = File(...),
    steps: str = Form(...),
    x_api_key: str = Header(default=""),
):
    if not API_KEY:
        raise HTTPException(status_code=503, detail="not_configured")
    if not secrets.compare_digest(x_api_key.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="bad_key")

    # Read and check the plan.
    try:
        plan = json.loads(steps)
        assert isinstance(plan, list) and 1 <= len(plan) <= 6
        for st in plan:
            assert st["action"] in ("straight", "turn_left", "turn_right", "blink")
            assert isinstance(st["count"], int) and 1 <= st["count"] <= 16
        assert plan[0]["action"] == "straight"
        assert sum(st["count"] for st in plan) == len(frames) <= MAX_FRAMES
    except Exception:
        raise HTTPException(status_code=400, detail="bad_plan")

    arrays = []
    for f in frames:
        data = f.file.read(FRAME_MAX_BYTES + 1)
        if len(data) > FRAME_MAX_BYTES:
            raise HTTPException(status_code=413, detail="too_large")
        arrays.append(_load(data, LIVE_SIDE))

    if not _slots.acquire(timeout=60):
        raise HTTPException(status_code=503, detail="busy")
    try:
        measures = [_measure(a) for a in arrays]

        def encode(i):
            m = measures[i]
            enc = face_recognition.face_encodings(arrays[i], [m["loc"]]) if m else []
            return enc[0] if enc else None

        result = _decide(plan, measures, encode)
        result["frames_with_face"] = sum(1 for m in measures if m)
        return result
    finally:
        _slots.release()

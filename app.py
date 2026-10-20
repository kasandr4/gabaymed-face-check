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

Lower "distance" = faces are closer. Defaults: <= 0.50 match, <= 0.62 unclear.
Change them with the MATCH_MAX / UNCLEAR_MAX settings after testing with real
sign-ups.
"""
import io
import os
import secrets
import threading

import face_recognition
import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

API_KEY = os.environ.get("API_KEY", "")
MATCH_MAX = float(os.environ.get("MATCH_MAX", "0.50"))
UNCLEAR_MAX = float(os.environ.get("UNCLEAR_MAX", "0.62"))
MAX_BYTES = int(os.environ.get("MAX_BYTES", str(6 * 1024 * 1024)))
MAX_SIDE = int(os.environ.get("MAX_SIDE", "800"))   # keeps memory under 512 MB (free hosts)

app = FastAPI(title="GabayMed face check", docs_url=None, redoc_url=None, openapi_url=None)

# The free host has 512 MB of memory: handle one comparison at a time.
_slots = threading.BoundedSemaphore(1)


def _load(data: bytes) -> np.ndarray:
    """Decode an upload to an RGB array, honouring phone rotation, size-capped."""
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="unreadable_image")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
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

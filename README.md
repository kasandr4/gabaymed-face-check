# GabayMed face check

Small helper for the GabayMed sign-up ID review. It never approves or rejects
anyone: it only gives the Front Desk hints. Photos are processed in memory only
(never stored or logged).

Set the `API_KEY` environment variable on the host.

## Endpoints (header `X-Api-Key` on all but `/health`)

- `GET /health`
- `POST /compare`  fields `selfie`, `id_image` -> how close the two faces are
- `POST /liveness` fields `frames` (several images in order), `steps` (JSON, e.g.
  `[{"action":"straight","count":4},{"action":"turn_right","count":6},{"action":"blink","count":12}]`)
  -> checks that a face is visible, the head really turned the asked way, the
  eyes really closed, and it is the same face throughout.

## Settings (all optional environment variables)

| Name | Default | What it does |
|---|---|---|
| `MATCH_MAX` / `UNCLEAR_MAX` | 0.50 / 0.62 | face-distance limits for `/compare` |
| `TURN_MIN_SHIFT` | 0.10 | how far the nose must slide for a head turn to count |
| `BLINK_MIN_DROP` / `BLINK_MAX_RATIO` | 0.07 / 0.75 | how much the eyes must close for a blink to count |
| `SAME_PERSON_MAX` | 0.62 | largest face distance allowed between steps |
| `TURN_FLIP` | 0 | set to `1` if "turn right" is reversed on real phones |
| `MAX_SIDE` / `LIVE_SIDE` | 800 / 480 | photo sizes used (keeps memory under 512 MB) |

These are starting values. Tune them by trying real people on real phones.

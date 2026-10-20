# GabayMed face check

Small helper for the GabayMed sign-up ID review. It compares a selfie with the
photo on an ID and returns how close the two faces are. It is only a hint for
the Front Desk: it never approves or rejects anyone, and photos are processed
in memory only (never stored or logged).

Set the `API_KEY` environment variable on the host. Optional settings:
`MATCH_MAX` (default 0.50), `UNCLEAR_MAX` (default 0.62), `MAX_SIDE` (default 800).

Endpoints: `GET /health`, `POST /compare` (header `X-Api-Key`, fields `selfie`, `id_image`).

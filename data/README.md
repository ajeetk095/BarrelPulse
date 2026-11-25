# Data
`fred/` holds a CSV snapshot of five FRED series (WTISPLC, INDPRO, CPIENGSL, DTWEXBGS, VIXCLS) with checksums in `MANIFEST.json`. The retrieval date of the original files is unknown; the manifest records checksums and the last observation per series. `cache/` is created at runtime (git-ignored). Refresh with `python scripts/refresh_snapshot.py` on a machine with internet access.

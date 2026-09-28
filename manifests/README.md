# Dataset layout and provenance

These six JSONL files record the exact image IDs, partitions, relative paths, and SHA-256 hashes used for the formal runs. Only `*_official.jsonl` manifests are included; earlier engineering splits were excluded. The NJU2K and NLPR training archives provide the 2,185-image optimization pool; the corresponding official test archives and SIP are held out. Each `paths` entry is relative to the repository root. The source archives must be acquired separately under their provider terms. If a provider distributes a different folder layout, update only the path fields after verifying each image hash.

The 2026 split is deterministic. `src/build_official_manifests.py` recreates it from the downloaded archive layout and computes image hashes. The executed fitting, validation, and calibration counts are 1,749, 218, and 218.

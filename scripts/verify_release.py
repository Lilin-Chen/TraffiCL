"""Verify the unchanged implementation files against completed V5 runs."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
expected = json.loads((root / "reproducibility/implementation_sha256.json").read_text())
bad = [name for name, digest in expected.items()
       if not (root / name).is_file()
       or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest]
if bad:
    raise SystemExit("Implementation verification failed: " + ", ".join(bad))
print(f"PASS: {len(expected)} implementation files match the recorded V5 runs.")

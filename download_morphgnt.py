"""Download the MorphGNT SBLGNT files for Matthew, Mark and Luke, pinned to one commit.

MorphGNT: Tauber, J. K., ed. (2017) MorphGNT: SBLGNT Edition. Version 6.12
[Data set]. https://github.com/morphgnt/sblgnt  DOI: 10.5281/zenodo.376200.
Morphological parsing and lemmatisation: CC BY-SA 3.0; the SBLGNT text itself
is subject to the SBLGNT EULA (https://sblgnt.com/license/).

The files are not redistributed with this repository; this script fetches
them from the pinned commit and verifies their SHA-256 checksums.
"""
import hashlib
import pathlib
import sys

import requests

COMMIT = "aaed91e57c8e4a8dc9a2383e129ca5e75fe6393d"
BASE = f"https://raw.githubusercontent.com/morphgnt/sblgnt/{COMMIT}"
FILES = {
    "61-Mt-morphgnt.txt": "95b360228bedd1c2d06655469111d3f028a4ce1d46939d440ed549feee8b73ec",
    "62-Mk-morphgnt.txt": "4a2d2365cfe3e8d6c20b7c130c2c46363fa4459c3a6004992eba0a18231b183e",
    "63-Lk-morphgnt.txt": "a6f10ccf27a905f21ecbec96c3c01f220c39dd37aae118022113a52dfeb7528e",
}
OUT = pathlib.Path(__file__).parent / "data" / "raw" / "morphgnt"


def download(name: str, sha256: str) -> None:
    print(f"  Downloading {name} ...", flush=True)
    r = requests.get(f"{BASE}/{name}", timeout=60)
    r.raise_for_status()
    digest = hashlib.sha256(r.content).hexdigest()
    if digest != sha256:
        sys.exit(f"Checksum mismatch for {name}: {digest} (expected {sha256})")
    (OUT / name).write_bytes(r.content)
    print(f"  Saved {len(r.content.splitlines())} words -> {OUT / name}", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MorphGNT SBLGNT (commit {COMMIT[:7]}) ...")
    for name, sha in FILES.items():
        download(name, sha)
    print("Done.")

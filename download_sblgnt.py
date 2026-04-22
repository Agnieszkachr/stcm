"""Download SBLGNT text files from LogosBible/SBLGNT GitHub repository."""
import requests
import pathlib
import sys

BASE = "https://raw.githubusercontent.com/LogosBible/SBLGNT/master/data/sblgnt/text"
BOOKS = {"Matt": "matthew", "Mark": "mark", "Luke": "luke"}
OUT = pathlib.Path(__file__).parent / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

def download(book_abbr: str, canonical: str) -> None:
    url = f"{BASE}/{book_abbr}.txt"
    print(f"  Downloading {book_abbr}.txt ...", flush=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    raw = r.content.decode("utf-8")
    out_path = OUT / f"{canonical}.txt"
    out_path.write_text(raw, encoding="utf-8")
    lines = raw.strip().splitlines()
    print(f"  Saved {len(lines)} verses -> {out_path}", flush=True)

if __name__ == "__main__":
    print("Downloading SBLGNT synoptic gospels ...")
    for abbr, name in BOOKS.items():
        download(abbr, name)
    print("Done.")

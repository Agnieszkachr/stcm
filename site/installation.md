# Installation

## Prerequisites

- Python 3.11, 3.12, or 3.13
- Git
- [uv](https://docs.astral.sh/uv/) (recommended) — or pip

## Step-by-Step

### 1. Clone the repository

```bash
git clone https://github.com/stcm-research/stcm.git
cd stcm
```

### 2. Create a virtual environment and install dependencies

**With uv (recommended):**

```bash
uv venv
uv pip install -r requirements.txt
```

**Or with pip:**

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Download the SBLGNT text

The SBL Greek New Testament text is required. It will be downloaded from the Logos Bible Software GitHub repository.

```bash
python download_sblgnt.py
```

This places `matthew.txt`, `mark.txt`, and `luke.txt` in `data/raw/`.

You are now ready to run the STCM pipeline.

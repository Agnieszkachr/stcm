# Installation

## Prerequisites

- Python 3.11, 3.12, or 3.13
- Git
- [uv](https://docs.astral.sh/uv/) (recommended) — or pip

## Step-by-Step

### 1. Clone the repository

```bash
git clone https://github.com/Agnieszkachr/stcm.git
cd stcm
```

### 2. Create the environment and install dependencies

**With uv (recommended):**

```bash
uv sync --locked
```

This creates `.venv` with the exact versions recorded in `uv.lock` (including transformers 4.57.3, used for the published numbers) and the test dependencies. Run every script through `uv run`, e.g. `uv run advanced_analysis.py`; no manual activation is needed.

**Or with pip:**

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

With pip, run the scripts with `python` instead of `uv run`.

### 3. Download the SBLGNT text

The SBL Greek New Testament text is required. It will be downloaded from the Logos Bible Software GitHub repository.

```bash
uv run download_sblgnt.py
```

This places `matthew.txt`, `mark.txt`, and `luke.txt` in `data/raw/`.

You are now ready to run the STCM pipeline.

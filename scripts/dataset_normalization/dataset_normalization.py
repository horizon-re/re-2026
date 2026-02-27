#!/usr/bin/env python3
"""
Dataset Normalization Script
----------------------------

Normalizes heterogeneous software requirements datasets into a single
canonical CSV format:

sample_id,text,label,source,domain,granularity,notes
"""
from pathlib import Path
import pandas as pd
from typing import List, Dict


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "11_extra_requirements"
OUTPUT_DIR = INPUT_DIR / "normalized"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "requirements_normalized.csv"

# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    return (
        str(text)
        .strip()
        .strip('"')
        .strip("'")
        .replace("\n", " ")
        .replace("  ", " ")
    )

def make_sample_id(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx:06d}"

# ---------------------------------------------------------------------
# DATASET ADAPTERS
# ---------------------------------------------------------------------

def load_dosspre(path: Path) -> List[Dict]:
    df = pd.read_csv(path)

    rows = []
    for i, r in df.iterrows():
        rows.append({
            "sample_id": make_sample_id("dosspre", i + 1),
            "text": clean_text(r["Requirement"]),
            "label": r["Class"],
            "source": "DoSSPRe",
            "domain": "fintech",
            "granularity": "sentence",
            "notes": ""
        })
    return rows

def load_fnfc(path: Path) -> List[Dict]:
    df = pd.read_csv(path)

    rows = []
    for i, r in df.iterrows():
        rows.append({
            "sample_id": make_sample_id("fnfc", i + 1),
            "text": clean_text(r["text"]),
            "label": r["class"],
            "source": "FNFC",
            "domain": "industrial",
            "granularity": "sentence",
            "notes": ""
        })
    return rows

def load_kaggle_extended(path: Path) -> List[Dict]:
    df = pd.read_csv(path)

    rows = []
    for i, r in df.iterrows():
        rows.append({
            "sample_id": make_sample_id("kaggle", i + 1),
            "text": clean_text(r["Requirement"]),
            "label": r["Type"],
            "source": "Kaggle",
            "domain": "unknown",
            "granularity": "sentence",
            "notes": ""
        })
    return rows

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    all_rows = []

    print("📥 Loading DOSSPRE...")
    all_rows.extend(
        load_dosspre(INPUT_DIR / "DOSSPRE 1.0Original.csv")
    )

    print("📥 Loading FNFC...")
    all_rows.extend(
        load_fnfc(INPUT_DIR / "HF_FNFC_Functional_Non-Functional_Classification.csv")
    )

    print("📥 Loading Kaggle Extended...")
    all_rows.extend(
        load_kaggle_extended(INPUT_DIR / "kaggle_software_requirements_extended.csv")
    )

    df = pd.DataFrame(all_rows)

    print(f"✅ Total normalized samples: {len(df)}")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"📦 Saved → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

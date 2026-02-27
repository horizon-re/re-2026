#!/usr/bin/env python3
"""
MPNet Training Dataset Merger
--------------------------------------

Merges and normalizes sentence-level samples for MPNet fine-tuning
(requirement vs non-requirement).

Sources:
- Synthetic non-requirements (JSONL)
- External normalized requirements (CSV)
- req_pipeline mixed dataset (JSONL with llm_labels)

Output:
- classifier/datasets/mpnet/train_merged.jsonl
"""

from __future__ import annotations
import argparse, csv, json, os, sys, re, hashlib, time
from pathlib import Path
from typing import Dict, Any, List, Set

# ---------------------------------------------------------------------
# SMART ROOT DETECTION
# ---------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

print(f"[init] project root: {PROJECT_ROOT}")

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
SYNTHETIC_NR = Path("classifier/synthetic/non_requirements/synthetic_non_requirements.jsonl")
EXTRA_REQS = Path("11_extra_requirements/normalized/requirements_normalized.csv")
req_pipeline_TRAIN = Path("classifier/outputs/splits/train_40.jsonl")

OUT_DIR = Path("classifier/datasets/mpnet")
OUT_FILE = OUT_DIR / "train_merged.jsonl"

MIN_TOKENS = 6
MAX_TOKENS = 128

# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def normalize_text(text) -> str:
    if not isinstance(text, str):
        return ""

    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_len(text: str) -> int:
    return len(text.split())

def valid_sentence(text: str) -> bool:
    t = token_len(text)
    return MIN_TOKENS <= t <= MAX_TOKENS

def make_uid(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------
# LOADERS
# ---------------------------------------------------------------------
def load_synthetic_nonreq(seen: Set[str]) -> List[Dict[str, Any]]:
    rows = []
    with open(SYNTHETIC_NR, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = normalize_text(obj.get("text", ""))
            if not valid_sentence(text):
                continue

            uid = make_uid(text)
            if uid in seen:
                continue
            seen.add(uid)

            rows.append({
                "uid": uid,
                "text": text,
                "label": 0,
                "source": "synthetic",
                "domain": "generic",
                "granularity": "sentence",
                "meta": {
                    "origin_id": obj.get("id"),
                    "model": obj.get("model")
                }
            })
    return rows

def load_extra_requirements(seen: Set[str]) -> List[Dict[str, Any]]:
    rows = []
    with open(EXTRA_REQS, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            text = normalize_text(r.get("text", ""))
            if not valid_sentence(text):
                continue

            uid = make_uid(text)
            if uid in seen:
                continue
            seen.add(uid)

            rows.append({
                "uid": uid,
                "text": text,
                "label": 1,
                "source": r.get("source", "external"),
                "domain": r.get("domain", "unknown"),
                "granularity": r.get("granularity", "sentence"),
                "meta": {
                    "origin_id": r.get("sample_id")
                }
            })
    return rows

def load_req_pipeline_mixed(seen: Set[str]) -> List[Dict[str, Any]]:
    rows = []
    with open(req_pipeline_TRAIN, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            labels = obj.get("llm_labels", [])

            if "requirement" in labels:
                label = 1
            elif "non_requirement" in labels:
                label = 0
            else:
                continue

            text = normalize_text(obj.get("sentence", ""))
            if not valid_sentence(text):
                continue

            uid = make_uid(text)
            if uid in seen:
                continue
            seen.add(uid)

            rows.append({
                "uid": uid,
                "text": text,
                "label": label,
                "source": "req_pipeline",
                "domain": obj.get("domain", "unknown"),
                "granularity": "sentence",
                "meta": {
                    "origin_id": obj.get("sent_id"),
                    "confidence": obj.get("confidence"),
                    "annotated_by": obj.get("annotated_by")
                }
            })
    return rows

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    t0 = time.time()
    ensure_dir(OUT_DIR)

    seen: Set[str] = set()
    merged: List[Dict[str, Any]] = []

    print("[load] synthetic non-requirements")
    merged += load_synthetic_nonreq(seen)

    print("[load] external requirements")
    merged += load_extra_requirements(seen)

    print("[load] req_pipeline mixed dataset")
    merged += load_req_pipeline_mixed(seen)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"[done] merged samples: {len(merged)}")
    print(f"[done] output: {OUT_FILE}")
    print(f"[done] time: {elapsed:.1f}s")
    print("=" * 60)

if __name__ == "__main__":
    main()

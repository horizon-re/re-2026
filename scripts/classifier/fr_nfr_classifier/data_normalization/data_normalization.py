#!/usr/bin/env python3
"""
Plantric — FR/NFR Data Normalization + Split Builder
----------------------------------------------------
Inputs:
  1) 11_extra_requirements/DOSSPRE 1.0Original.csv
  2) 11_extra_requirements/HF_FNFC_Functional_Non-Functional_Classification.csv
  3) 11_extra_requirements/kaggle_software_requirements_extended.csv
  4) classifier/outputs/by_category/*.jsonl

Rules:
  - Datasets 1–3: take 80% each into pool
  - Dataset 4: take 40% total into pool
  - Map to labels: F vs NFR
  - Merge pool, dedupe, stratified split into train/dev/test (80/10/10)

Outputs:
  classifier/outputs/splits_frnfr/
    train.jsonl
    dev.jsonl
    test.jsonl
    stats.json
    config.json
"""

from __future__ import annotations
import os, sys, json, time, argparse, random, hashlib, csv, re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# ---------------------------------------------------------------------
# ROOT
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
print(f"[init] project root: {PROJECT_ROOT}")

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
P1 = Path("11_extra_requirements/DOSSPRE 1.0Original.csv")
P2 = Path("11_extra_requirements/HF_FNFC_Functional_Non-Functional_Classification.csv")
P3 = Path("11_extra_requirements/kaggle_software_requirements_extended.csv")
P4_DIR = Path("classifier/outputs/by_category")

OUT_DIR = Path("classifier/outputs/splits_frnfr")
CACHE_DIR = Path("classifier/datasets/normalized/cache_frnfr")

# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def now_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def sha1_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def clean_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def norm_key(s: str) -> str:
    # exact-ish dedupe key
    return clean_text(s).lower()

def label_frnfr_from_ds123(raw: str) -> str:
    r = (raw or "").strip().lower()
    if r in ["f", "functional", "functionality"]:
        return "F"
    return "NFR"

def label_frnfr_from_own(llm_labels: List[str]) -> str:
    llm_labels = llm_labels or []
    return "F" if "functional" in llm_labels else "NFR"

def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        return list(csv.DictReader(f))

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def sample_ratio(rows: List[Dict[str, Any]], ratio: float) -> List[Dict[str, Any]]:
    k = int(len(rows) * ratio)
    if k <= 0:
        return []
    return random.sample(rows, k)

def stratified_split(rows: List[Dict[str, Any]], seed: int) -> Tuple[List, List, List]:
    y = np.array([1 if r["label"] == "F" else 0 for r in rows])
    idx = np.arange(len(rows))

    # First split: train vs temp
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, temp_idx = next(sss1.split(idx, y))

    # Second split: dev vs test from temp (0.1/0.1 overall)
    y_temp = y[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    dv_rel, ts_rel = next(sss2.split(np.arange(len(temp_idx)), y_temp))

    train = [rows[i] for i in tr_idx]
    dev = [rows[temp_idx[i]] for i in dv_rel]
    test = [rows[temp_idx[i]] for i in ts_rel]
    return train, dev, test

def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    f = sum(1 for r in rows if r["label"] == "F")
    nfr = n - f
    return {"n": n, "F": f, "NFR": nfr, "F_ratio": (f / n if n else 0.0)}

# ---------------------------------------------------------------------
# LOADERS
# ---------------------------------------------------------------------
def load_ds1() -> List[Dict[str, Any]]:
    rows = read_csv(P1)
    out = []
    for i, r in enumerate(rows):
        text = clean_text(r.get("Requirement", ""))
        if not text:
            continue
        out.append({
            "id": f"dosspre_{i}",
            "text": text,
            "label": label_frnfr_from_ds123(r.get("Class", "")),
            "source": "dosspre",
            "context": False
        })
    return out

def load_ds2() -> List[Dict[str, Any]]:
    rows = read_csv(P2)
    out = []
    for i, r in enumerate(rows):
        text = clean_text(r.get("text", ""))
        if not text:
            continue
        out.append({
            "id": f"hf_{i}",
            "text": text,
            "label": label_frnfr_from_ds123(r.get("class", "")),
            "source": "hf",
            "context": False
        })
    return out

def load_ds3() -> List[Dict[str, Any]]:
    rows = read_csv(P3)
    out = []
    for i, r in enumerate(rows):
        text = clean_text(r.get("Requirement", ""))
        if not text:
            continue
        out.append({
            "id": f"kaggle_{i}",
            "text": text,
            "label": label_frnfr_from_ds123(r.get("Type", "")),
            "source": "kaggle",
            "context": False
        })
    return out

def load_ds4() -> List[Dict[str, Any]]:
    files = sorted(P4_DIR.glob("*.jsonl"))
    if not files:
        raise RuntimeError(f"No jsonl files found in {P4_DIR}")
    out = []
    for fp in files:
        rows = read_jsonl(fp)
        for r in rows:
            text = clean_text(r.get("sentence", ""))
            labels = r.get("llm_labels", []) or []
            if not text:
                continue
            out.append({
                "id": r.get("sent_id") or f"own_{len(out)}",
                "text": text,
                "label": label_frnfr_from_own(labels),
                "source": "own",
                "context": ("with_context" in labels)
            })
    return out

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratio-ds123", type=float, default=0.8)
    ap.add_argument("--ratio-own", type=float, default=0.4)
    ap.add_argument("--dedupe", action="store_true", default=True)
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(OUT_DIR)
    ensure_dir(CACHE_DIR)

    run_id = now_run_id()
    cfg = {
        "run_id": run_id,
        "seed": args.seed,
        "ratio_ds123": args.ratio_ds123,
        "ratio_own": args.ratio_own,
        "inputs": [str(P1), str(P2), str(P3), str(P4_DIR)],
        "dedupe": bool(args.dedupe),
        "split": {"train": 0.8, "dev": 0.1, "test": 0.1},
    }

    cache_key = sha1_str(json.dumps(cfg, sort_keys=True))
    cache_path = CACHE_DIR / f"pool_{cache_key}.json"

    # Build or load merged pool
    if cache_path.exists():
        print(f"[cache] loading pool from {cache_path.name}")
        pool = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        ds1 = load_ds1()
        ds2 = load_ds2()
        ds3 = load_ds3()
        ds4 = load_ds4()

        s1 = sample_ratio(ds1, args.ratio_ds123)
        s2 = sample_ratio(ds2, args.ratio_ds123)
        s3 = sample_ratio(ds3, args.ratio_ds123)
        s4 = sample_ratio(ds4, args.ratio_own)

        pool = s1 + s2 + s3 + s4
        random.shuffle(pool)

        # Dedupe (exact-ish)
        if args.dedupe:
            seen = set()
            deduped = []
            for r in pool:
                k = norm_key(r["text"])
                if k in seen:
                    continue
                seen.add(k)
                deduped.append(r)
            pool = deduped

        cache_path.write_text(json.dumps(pool, ensure_ascii=False), encoding="utf-8")
        print(f"[cache] saved pool to {cache_path.name}")

    # Split
    train, dev, test = stratified_split(pool, seed=args.seed)

    # Write
    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "dev.jsonl", dev)
    write_jsonl(OUT_DIR / "test.jsonl", test)

    stats = {
        "pool": summarize(pool),
        "train": summarize(train),
        "dev": summarize(dev),
        "test": summarize(test),
    }

    (OUT_DIR / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("[done] wrote splits:")
    print(f"  - {OUT_DIR/'train.jsonl'} ({stats['train']})")
    print(f"  - {OUT_DIR/'dev.jsonl'} ({stats['dev']})")
    print(f"  - {OUT_DIR/'test.jsonl'} ({stats['test']})")

if __name__ == "__main__":
    main()

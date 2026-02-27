#!/usr/bin/env python3
"""
Step 3 — Atomic Requirement Unit Splitter (Dual Mode)
-----------------------------------------------------
Generates atomic requirement units using:
- Conservative atomicity
- Aggressive atomicity

Exports per-requirement and global matrices for analysis.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import spacy

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
STEP2_DIR = ROOT / "classifier" / "step_2"
STEP3_DIR = ROOT / "classifier" / "step_3"
MODES = ["conservative", "aggressive"]

NLP = spacy.load("en_core_web_sm")

SEQUENTIAL_MARKERS = {"then", "after", "before", "once", "subsequently"}

# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------
def write_jsonl(path: Path, rows: List[Dict]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_unit_id(req_id: str, mode: str, idx: int) -> str:
    return f"{req_id}::{mode[0]}u{idx:03d}"


# ---------------------------------------------------------------------
# VERB EXTRACTION
# ---------------------------------------------------------------------
def extract_main_verbs(doc):
    return [t for t in doc if t.pos_ == "VERB" and t.dep_ in {"ROOT", "conj"}]


# ---------------------------------------------------------------------
# SPLIT STRATEGIES
# ---------------------------------------------------------------------
def conservative_split(doc) -> List[str]:
    verbs = extract_main_verbs(doc)

    if len(verbs) <= 1 and not any(t.text.lower() in SEQUENTIAL_MARKERS for t in doc):
        return [doc.text.strip()]

    units = []
    for v in verbs:
        subtree = sorted(list(v.subtree), key=lambda t: t.i)
        units.append(" ".join(t.text for t in subtree).strip())
    return units


def aggressive_split(doc) -> List[str]:
    units = []
    for tok in doc:
        if tok.pos_ == "VERB":
            subtree = sorted(list(tok.subtree), key=lambda t: t.i)
            text = " ".join(t.text for t in subtree).strip()
            if len(text.split()) > 2:
                units.append(text)
    return units if units else [doc.text.strip()]


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    t0 = time.time()

    global_stats = {
        "conservative": defaultdict(int),
        "aggressive": defaultdict(int),
    }

    for mode in MODES:
        for domain_dir in STEP2_DIR.iterdir():
            if not domain_dir.is_dir():
                continue

            for prompt_dir in domain_dir.iterdir():
                if not prompt_dir.is_dir():
                    continue

                for req_dir in prompt_dir.iterdir():
                    cand_path = req_dir / "candidates.jsonl"
                    if not cand_path.exists():
                        continue

                    # records = [json.loads(l) for l in cand_path.read_text().splitlines()]
                    records = [
                        json.loads(l)
                        for l in cand_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    ]  

                    if not records:
                        continue

                    out_dir = STEP3_DIR / mode / domain_dir.name / prompt_dir.name / req_dir.name
                    out_dir.mkdir(parents=True, exist_ok=True)

                    atomic_units = []
                    split_trace = []

                    unit_idx = 1
                    max_from_one = 1
                    split_sentences = 0

                    for rec in records:
                        doc = NLP(rec["text"])
                        parts = (
                            conservative_split(doc)
                            if mode == "conservative"
                            else aggressive_split(doc)
                        )

                        if len(parts) > 1:
                            split_sentences += 1
                            max_from_one = max(max_from_one, len(parts))
                            split_trace.append({
                                "parent_sent_id": rec["sent_id"],
                                "mode": mode,
                                "num_units": len(parts),
                                "original_text": rec["text"],
                                "units": parts
                            })

                        for part in parts:
                            atomic_units.append({
                                "unit_id": make_unit_id(rec["req_id"], mode, unit_idx),
                                "req_id": rec["req_id"],
                                "doc_id": rec["doc_id"],
                                "parent_sent_id": rec["sent_id"],
                                "order": unit_idx,
                                "text": part
                            })
                            unit_idx += 1

                    # --- Metrics
                    num_parent = len(records)
                    num_units = len(atomic_units)

                    atomic_matrix = {
                        "req_id": req_dir.name,
                        "mode": mode,
                        "num_parent_sentences": num_parent,
                        "num_atomic_units": num_units,
                        "avg_units_per_sentence": round(num_units / max(1, num_parent), 3),
                        "max_units_from_single_sentence": max_from_one,
                        "split_rate": round(split_sentences / max(1, num_parent), 3)
                    }

                    # --- Write outputs
                    write_jsonl(out_dir / "atomic_units.jsonl", atomic_units)
                    write_jsonl(out_dir / "split_trace.jsonl", split_trace)
                    (out_dir / "atomic_matrix.json").write_text(
                        json.dumps(atomic_matrix, indent=2),
                        encoding="utf-8"
                    )

                    # --- Global stats
                    global_stats[mode]["total_units"] += num_units
                    global_stats[mode]["total_reqs"] += 1
                    global_stats[mode]["total_sentences"] += num_parent
                    global_stats[mode]["total_splits"] += split_sentences

    # -----------------------------------------------------------------
    # GLOBAL COMPARISON
    # -----------------------------------------------------------------
    comparison = {}
    for mode, s in global_stats.items():
        comparison[mode] = {
            "total_units": s["total_units"],
            "avg_units_per_req": round(s["total_units"] / max(1, s["total_reqs"]), 3),
            "split_rate": round(s["total_splits"] / max(1, s["total_sentences"]), 3)
        }

    (STEP3_DIR / "_atomicity_comparison.json").write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8"
    )

    print(f"[done] Step 3 complete (dual-mode) in {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()

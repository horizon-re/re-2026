#!/usr/bin/env python3
"""
Step 3a — Local Requirement Candidate Classifier
------------------------------------------------
Classifies atomic units as:
- accepted
- rejected
- borderline

Uses ONLY local text (no context).
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List
import spacy
import math

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
STEP3_DIR = ROOT / "classifier" / "step_3"
STEP3A_DIR = ROOT / "classifier" / "step_3a"
STEP3A_DIR.mkdir(parents=True, exist_ok=True)

ACCEPT_THR = 0.70
REJECT_THR = 0.30

NLP = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

MODALS = {"shall", "must", "should", "will", "may"}

# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------
def write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def compute_rule_features(text: str) -> Dict[str, float]:
    doc = NLP(text)

    tokens = [t for t in doc if not t.is_space]
    verbs = [t for t in doc if t.pos_ == "VERB"]
    subjects = [t for t in doc if t.dep_ in {"nsubj", "nsubjpass"}]

    has_modal = any(t.text.lower() in MODALS for t in doc)

    return {
        "token_count": len(tokens),
        "has_verb": 1.0 if verbs else 0.0,
        "has_subject": 1.0 if subjects else 0.0,
        "has_modal": 1.0 if has_modal else 0.0,
        "is_fragment": 1.0 if len(tokens) <= 2 else 0.0,
    }


def score_candidate(feat: Dict[str, float]) -> float:
    """
    Conservative linear scoring.
    (Classifier slot — replace later with trained LR)
    """
    score = 0.0
    score += 0.35 * feat["has_verb"]
    score += 0.25 * feat["has_subject"]
    score += 0.25 * feat["has_modal"]
    score += 0.15 * (1.0 if feat["token_count"] >= 4 else 0.0)
    score -= 0.20 * feat["is_fragment"]

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    global_metrics = {
        "total": 0,
        "accepted": 0,
        "rejected": 0,
        "borderline": 0,
    }

    for domain_dir in STEP3_DIR.iterdir():
        if not domain_dir.is_dir():
            continue

        for prompt_dir in domain_dir.iterdir():
            if not prompt_dir.is_dir():
                continue

            for req_dir in prompt_dir.iterdir():
                atomic_path = req_dir / "atomic_units.jsonl"
                if not atomic_path.exists():
                    continue

                units = [json.loads(l) for l in atomic_path.read_text(encoding="utf-8").splitlines()]
                if not units:
                    continue

                accepted, rejected, borderline = [], [], []

                for u in units:
                    feats = compute_rule_features(u["text"])
                    score = score_candidate(feats)

                    record = {
                        **u,
                        "candidate_score": round(score, 3),
                        "features": feats,
                    }

                    global_metrics["total"] += 1

                    if score >= ACCEPT_THR:
                        accepted.append(record)
                        global_metrics["accepted"] += 1
                    elif score <= REJECT_THR:
                        rejected.append(record)
                        global_metrics["rejected"] += 1
                    else:
                        borderline.append(record)
                        global_metrics["borderline"] += 1

                out_dir = STEP3A_DIR / domain_dir.name / prompt_dir.name / req_dir.name
                write_jsonl(out_dir / "accepted.jsonl", accepted)
                write_jsonl(out_dir / "rejected.jsonl", rejected)
                write_jsonl(out_dir / "borderline.jsonl", borderline)

                metrics = {
                    "req_id": req_dir.name,
                    "accepted": len(accepted),
                    "rejected": len(rejected),
                    "borderline": len(borderline),
                    "total": len(units),
                }
                (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"[done] Step 3a complete — {global_metrics}")


if __name__ == "__main__":
    main()

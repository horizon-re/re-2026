#!/usr/bin/env python3
"""
req_pipeline — Phase 1 Holdout Evaluation
------------------------------------

Evaluates the sentence-only Phase-1 MPNet requirement classifier
on holdout data.

- Inference only (NO training, NO tuning)
- Auto-selects latest Phase-1 run
- Produces predictions, metrics, confusion matrix, and error slices

Outputs:
classifier/outputs/eval/phase1_<run_id>/
    - predictions.jsonl
    - metrics.json
    - confusion.json
    - errors_fp.jsonl
    - errors_fn.jsonl
"""

from __future__ import annotations
import os, sys, json
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

print(f"[init] project root: {PROJECT_ROOT}")

# ---------------------------------------------------------------------
# CONFIG (CHANGE HERE, NOT ALL OVER THE FILE)
# ---------------------------------------------------------------------
LIMIT = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HOLDOUT_PATH = Path("classifier/outputs/splits/holdout_60.jsonl")
MODEL_ROOT = Path("classifier/models/mpnet_phase1")
OUT_ROOT = Path("classifier/outputs/eval")

# ---------------------------------------------------------------------
# MODEL HEAD (must match Phase-1 training)
# ---------------------------------------------------------------------
class RequirementHead(nn.Module):
    def __init__(self, in_dim=768, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------
def load_jsonl(p: Path) -> List[Dict[str, Any]]:
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def mean_pool(hidden, mask):
    mask = mask.unsqueeze(-1).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

def map_gold(llm_labels: List[str]) -> int:
    return 1 if "requirement" in llm_labels else 0

def confidence_bucket(p: float) -> str:
    if p >= 0.85 or p <= 0.15:
        return "high"
    if p >= 0.65 or p <= 0.35:
        return "medium"
    return "low"

def get_latest_run_id(model_root: Path) -> str:
    runs = [d for d in model_root.iterdir() if d.is_dir()]
    if not runs:
        raise RuntimeError("No Phase-1 runs found.")
    runs.sort(key=lambda p: p.name)
    return runs[-1].name

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    device = torch.device(DEVICE)

    # -------- Select latest run --------
    run_id = get_latest_run_id(MODEL_ROOT)
    model_dir = MODEL_ROOT / run_id
    print(f"[auto] using latest Phase-1 run: {run_id}")

    ckpt = torch.load(model_dir / "best.pt", map_location=device)
    cfg = json.loads((model_dir / "config.json").read_text())

    out_dir = OUT_ROOT / f"phase1_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- Load model --------
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    encoder = AutoModel.from_pretrained(cfg["model"]).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    head = RequirementHead(
        in_dim=768,
        hidden=cfg["hidden"],
        dropout=cfg["dropout"],
    ).to(device)

    head.load_state_dict(ckpt["head_state"])
    head.eval()

    # -------- Load data --------
    rows = load_jsonl(HOLDOUT_PATH)[:LIMIT]
    print(f"[data] evaluating {len(rows)} samples")

    y_true, y_pred, y_prob = [], [], []
    predictions = []

    # -------- Inference --------
    with torch.no_grad():
        for r in rows:
            sent = r["sentence"].strip()
            gold = map_gold(r.get("llm_labels", []))

            enc = tokenizer(
                sent,
                truncation=True,
                padding=True,
                max_length=cfg["max_length"],
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            out = encoder(**enc)
            emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
            emb = nn.functional.normalize(emb, p=2, dim=1)

            logit = head(emb).item()
            prob = float(torch.sigmoid(torch.tensor(logit)).item())
            pred = int(prob >= 0.5)

            y_true.append(gold)
            y_pred.append(pred)
            y_prob.append(prob)

            predictions.append({
                "req_id": r["req_id"],
                "sent_id": r["sent_id"],
                "sentence": sent,
                "gold": gold,
                "pred": pred,
                "prob": round(prob, 4),
                "correct": int(pred == gold),
                "confidence_bucket": confidence_bucket(prob),
            })

    # -------- Metrics --------
    metrics = {}
    metrics["n"] = len(y_true)
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics.update({
        "precision": p,
        "recall": r,
        "f1": f1,
    })

    if len(set(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    cm = confusion_matrix(y_true, y_pred).tolist()
    metrics["confidence_buckets"] = dict(
        Counter(p["confidence_bucket"] for p in predictions)
    )

    # -------- Outputs --------
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "confusion.json", "w", encoding="utf-8") as f:
        json.dump({
            "labels": ["non_requirement", "requirement"],
            "matrix": cm,
        }, f, indent=2)

    with open(out_dir / "errors_fp.jsonl", "w", encoding="utf-8") as f:
        for p in predictions:
            if p["gold"] == 0 and p["pred"] == 1:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with open(out_dir / "errors_fn.jsonl", "w", encoding="utf-8") as f:
        for p in predictions:
            if p["gold"] == 1 and p["pred"] == 0:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # -------- Summary --------
    print("=" * 60)
    print("[done] Phase-1 holdout evaluation complete")
    for k, v in metrics.items():
        print(f"{k:>20}: {v}")
    print(f"[done] outputs → {out_dir}")
    print("=" * 60)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()

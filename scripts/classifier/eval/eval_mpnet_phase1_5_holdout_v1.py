#!/usr/bin/env python3
"""
Evaluate MPNet Phase-1 vs Phase-1.5 (contextual) on holdout data.

- Auto-detects input dims from checkpoints
- Pads missing engineered features for Phase-1.5
- Supports --limit
- Outputs full metrics
"""

from __future__ import annotations
import os, sys, json, argparse, time
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from transformers import AutoTokenizer, AutoModel


# ---------------------------------------------------------------------
# ROOT
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
print(f"[init] project root: {PROJECT_ROOT}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
PHASE1_ROOT = Path("classifier/models/mpnet_phase1")
PHASE15_ROOT = Path("classifier/models/mpnet_phase1_5_context")
HOLDOUT_PATH = Path("classifier/outputs/splits/holdout_60.jsonl")
OUT_ROOT = Path("classifier/outputs/eval")


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------
class RequirementHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.1):
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
# UTILITIES
# ---------------------------------------------------------------------
def load_jsonl(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def mean_pool(last_hidden, mask):
    mask = mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


@torch.no_grad()
def compute_embeddings(texts, tokenizer, encoder, batch_size=32):
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i:i + batch_size],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(DEVICE)
        h = encoder(**enc)
        pooled = mean_pool(h.last_hidden_state, enc["attention_mask"])
        pooled = nn.functional.normalize(pooled, p=2, dim=1)
        out.append(pooled.cpu())
    return torch.cat(out, dim=0)


def compute_metrics(y_true, probs):
    preds = (probs >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, preds),
        "precision": p,
        "recall": r,
        "f1": f1,
        "roc_auc": roc_auc_score(y_true, probs) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": average_precision_score(y_true, probs),
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / f"phase1_vs_phase1_5_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-select latest runs
    p1_dir = sorted(PHASE1_ROOT.iterdir())[-1]
    p15_dir = sorted(PHASE15_ROOT.iterdir())[-1]

    print(f"[auto] Phase-1:   {p1_dir.name}")
    print(f"[auto] Phase-1.5: {p15_dir.name}")

    ckpt1 = torch.load(p1_dir / "best.pt", map_location=DEVICE)
    ckpt15 = torch.load(p15_dir / "best.pt", map_location=DEVICE)

    dim1 = ckpt1["head_state"]["net.0.weight"].shape[1]
    dim15 = ckpt15["head_state"]["net.0.weight"].shape[1]

    h1 = RequirementHead(dim1).to(DEVICE)
    h1.load_state_dict(ckpt1["head_state"])
    h1.eval()

    hctx = RequirementHead(dim15).to(DEVICE)
    hctx.load_state_dict(ckpt15["head_state"])
    hctx.eval()

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    encoder = AutoModel.from_pretrained(
        "sentence-transformers/all-mpnet-base-v2"
    ).to(DEVICE)
    encoder.eval()

    rows = load_jsonl(HOLDOUT_PATH, args.limit)
    print(f"[data] evaluating {len(rows)} samples")

    sentences = [r["sentence"] for r in rows]
    ctx_texts = [
        " ".join(
            (r.get("context_before") or []) +
            [r["sentence"]] +
            (r.get("context_after") or [])
        )
        for r in rows
    ]

    y = np.array([
        1 if "requirement" in r.get("llm_labels", []) else 0
        for r in rows
    ])

    emb_sent = compute_embeddings(sentences, tokenizer, encoder)
    emb_ctx = compute_embeddings(ctx_texts, tokenizer, encoder)

    with torch.no_grad():
        # Phase-1
        p1_logits = h1(emb_sent.to(DEVICE)).cpu()
        p1_probs = torch.sigmoid(p1_logits).numpy()

        # Phase-1.5 feature reconstruction
        base_feat = torch.cat([emb_sent, emb_ctx], dim=1)
        cur_dim = base_feat.shape[1]
        exp_dim = dim15

        if cur_dim < exp_dim:
            pad = torch.zeros(
                (base_feat.shape[0], exp_dim - cur_dim),
                dtype=base_feat.dtype
            )
            x15 = torch.cat([base_feat, pad], dim=1)
            print(f"[info] padded Phase-1.5 features: {cur_dim} → {exp_dim}")
        elif cur_dim == exp_dim:
            x15 = base_feat
        else:
            raise RuntimeError(
                f"Phase-1.5 feature overflow: built {cur_dim}, expected {exp_dim}"
            )

        p15_logits = hctx(x15.to(DEVICE)).cpu()
        p15_probs = torch.sigmoid(p15_logits).numpy()

    m1 = compute_metrics(y, p1_probs)
    m15 = compute_metrics(y, p15_probs)

    summary = {
        "n": int(len(y)),
        "phase1": m1,
        "phase1_5": m15,
        "delta": {
            k: (m15[k] - m1[k]) if isinstance(m1[k], float) else None
            for k in m1
        }
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print("[done] Phase-1 vs Phase-1.5 evaluation")
    print(json.dumps(summary, indent=2))
    print("[done] outputs →", out_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()

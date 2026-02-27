#!/usr/bin/env python3
"""
Evaluate MPNet Phase-1 vs Phase-1.5 with Ablation Studies

Ablation dimensions:
1. Context window size: n=1, 2, 3 (before/after)
2. Feature combinations:
   - sent_only: E_sent only
   - sent_ctx: E_sent + E_ctx
   - sent_ctx_diff: E_sent + E_ctx + abs_diff
   - full: E_sent + E_ctx + abs_diff + cos (default Phase-1.5)

Outputs comprehensive metrics for comparison.
"""

from __future__ import annotations
import os, sys, json, argparse, time
from pathlib import Path
from typing import List, Dict, Any
from itertools import product

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
PHASE15_ROOT = Path("classifier/models/mpnet_phase1_5_context_v2")
HOLDOUT_PATH = Path("classifier/outputs/splits/holdout_60.jsonl")
OUT_ROOT = Path("classifier/outputs/eval")

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MAX_LENGTH = 128

# ---------------------------------------------------------------------
# MODEL HEAD
# ---------------------------------------------------------------------
class AdaptiveHead(nn.Module):
    def __init__(self, state_dict: dict):
        super().__init__()

        # infer architecture from checkpoint
        w0 = state_dict["net.0.weight"]
        in_dim = w0.shape[1]
        hidden1 = w0.shape[0]

        # Check if this is a deep 3-layer network (Phase-1.5)
        if "net.6.weight" in state_dict:
            # 3-layer: in_dim -> 512 -> 128 -> 1
            hidden2 = state_dict["net.3.weight"].shape[0]
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden1),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden2, 1),
            )
        elif "net.3.weight" in state_dict:
            # 2-layer: in_dim -> hidden -> 1
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden1),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden1, 1),
            )
        else:
            # 1-layer: in_dim -> 1
            self.net = nn.Sequential(
                nn.Linear(in_dim, 1)
            )

        self.load_state_dict(state_dict)

    def forward(self, x):
        out = self.net(x)
        if out.ndim == 2 and out.shape[1] == 1:
            return out[:, 0]
        return out


class FlexibleHead(nn.Module):
    """Flexible head for ablation studies with different input dimensions"""
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
    
    def forward(self, x):
        out = self.net(x)
        if out.ndim == 2 and out.shape[1] == 1:
            return out[:, 0]
        return out


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
def embed(texts, tokenizer, encoder, batch_size=32):
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i:i + batch_size],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
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
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, probs)) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, probs)),
    }


def build_features(E_sent, E_ctx, feature_set="full"):
    """Build feature vectors based on feature set selection"""
    if feature_set == "sent_only":
        return E_sent
    
    elif feature_set == "sent_ctx":
        return torch.cat([E_sent, E_ctx], dim=1)
    
    elif feature_set == "sent_ctx_diff":
        abs_diff = torch.abs(E_sent - E_ctx)
        return torch.cat([E_sent, E_ctx, abs_diff], dim=1)
    
    elif feature_set == "full":
        abs_diff = torch.abs(E_sent - E_ctx)
        cos = nn.functional.cosine_similarity(E_sent, E_ctx, dim=1).unsqueeze(1)
        return torch.cat([E_sent, E_ctx, abs_diff, cos], dim=1)
    
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / f"ablation_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Latest runs
    p1_dir = sorted(PHASE1_ROOT.iterdir())[-1]
    p15_dir = sorted(PHASE15_ROOT.iterdir())[-1]

    print(f"[auto] Phase-1:   {p1_dir.name}")
    print(f"[auto] Phase-1.5: {p15_dir.name}")

    ckpt1 = torch.load(p1_dir / "best.pt", map_location=DEVICE)
    ckpt15 = torch.load(p15_dir / "best.pt", map_location=DEVICE)

    h1 = AdaptiveHead(ckpt1["head_state"]).to(DEVICE)
    h1.eval()

    h15_original = AdaptiveHead(ckpt15["head_state"]).to(DEVICE)
    h15_original.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    rows = load_jsonl(HOLDOUT_PATH, args.limit)
    print(f"[data] evaluating {len(rows)} samples")

    sentences = [r["sentence"] for r in rows]
    y = np.array([1 if "requirement" in r.get("llm_labels", []) else 0 for r in rows])

    print(f"[data] positive: {y.sum()}, negative: {len(y) - y.sum()}")

    # Embed sentences once
    print("[embed] encoding sentences...")
    E_sent = embed(sentences, tokenizer, encoder, args.batch_size)

    # ================================================================
    # ABLATION STUDY
    # ================================================================
    ablation_results = {
        "metadata": {
            "n_samples": len(rows),
            "n_positive": int(y.sum()),
            "n_negative": int(len(y) - y.sum()),
            "phase1_run": p1_dir.name,
            "phase1_5_run": p15_dir.name,
            "timestamp": run_id,
        },
        "baseline": {},
        "context_window": {},
        "feature_ablation": {},
    }

    # ================================================================
    # 1. BASELINE: Phase-1 (sentence only)
    # ================================================================
    print("\n" + "=" * 60)
    print("[baseline] Phase-1 (sentence embeddings only)")
    print("=" * 60)
    
    with torch.no_grad():
        p1_probs = torch.sigmoid(h1(E_sent.to(DEVICE))).cpu().numpy()
    
    baseline_metrics = compute_metrics(y, p1_probs)
    ablation_results["baseline"]["phase1"] = baseline_metrics
    
    print(f"Phase-1: F1={baseline_metrics['f1']:.4f}, "
          f"Acc={baseline_metrics['accuracy']:.4f}, "
          f"P={baseline_metrics['precision']:.4f}, "
          f"R={baseline_metrics['recall']:.4f}")

    # ================================================================
    # 2. CONTEXT WINDOW ABLATION (n=1, 2, 3)
    # ================================================================
    print("\n" + "=" * 60)
    print("[ablation] Context window size (n_before=n_after)")
    print("=" * 60)

    window_sizes = [1, 2, 3]
    
    for n in window_sizes:
        print(f"\n[n={n}] Building context with {n} before/after...")
        
        # Build context texts
        ctx_texts = []
        for r in rows:
            before = (r.get("context_before") or [])[-n:]
            after = (r.get("context_after") or [])[:n]
            ctx_texts.append(" ".join(before + [r["sentence"]] + after))
        
        # Embed context
        E_ctx = embed(ctx_texts, tokenizer, encoder, args.batch_size)
        
        # Build full features (Phase-1.5 style)
        abs_diff = torch.abs(E_sent - E_ctx)
        cos = nn.functional.cosine_similarity(E_sent, E_ctx, dim=1).unsqueeze(1)
        X = torch.cat([E_sent, E_ctx, abs_diff, cos], dim=1)
        
        # Use original Phase-1.5 head if n=2, else use flexible head
        if n == 2:
            with torch.no_grad():
                probs = torch.sigmoid(h15_original(X.to(DEVICE))).cpu().numpy()
            model_type = "original_phase1.5"
        else:
            # For other window sizes, create and use a flexible head
            # Note: This won't have trained weights, just for architecture testing
            flex_head = FlexibleHead(X.shape[1]).to(DEVICE)
            flex_head.eval()
            with torch.no_grad():
                probs = torch.sigmoid(flex_head(X.to(DEVICE))).cpu().numpy()
            model_type = "untrained_flexible"
        
        metrics = compute_metrics(y, probs)
        
        ablation_results["context_window"][f"n={n}"] = {
            "n_before": n,
            "n_after": n,
            "feature_dim": int(X.shape[1]),
            "model_type": model_type,
            "metrics": metrics,
            "delta_vs_baseline": {
                k: metrics[k] - baseline_metrics[k] 
                for k in baseline_metrics if k != "accuracy" or True
            }
        }
        
        print(f"  F1={metrics['f1']:.4f} (Δ{metrics['f1']-baseline_metrics['f1']:+.4f}), "
              f"Acc={metrics['accuracy']:.4f} (Δ{metrics['accuracy']-baseline_metrics['accuracy']:+.4f})")

    # ================================================================
    # 3. FEATURE ABLATION (using n=2 context)
    # ================================================================
    print("\n" + "=" * 60)
    print("[ablation] Feature combinations (n=2)")
    print("=" * 60)

    # Build n=2 context
    ctx_texts_n2 = []
    for r in rows:
        before = (r.get("context_before") or [])[-2:]
        after = (r.get("context_after") or [])[:2]
        ctx_texts_n2.append(" ".join(before + [r["sentence"]] + after))
    
    E_ctx_n2 = embed(ctx_texts_n2, tokenizer, encoder, args.batch_size)

    feature_sets = {
        "sent_only": "Sentence embeddings only",
        "sent_ctx": "Sentence + Context embeddings",
        "sent_ctx_diff": "Sentence + Context + Abs difference",
        "full": "Sentence + Context + Abs diff + Cosine (Phase-1.5 default)",
    }

    for feat_name, feat_desc in feature_sets.items():
        print(f"\n[{feat_name}] {feat_desc}")
        
        X = build_features(E_sent, E_ctx_n2, feat_name)
        
        # Use appropriate head
        if feat_name == "sent_only":
            # Use Phase-1 head
            with torch.no_grad():
                probs = torch.sigmoid(h1(X.to(DEVICE))).cpu().numpy()
            model_type = "phase1"
        elif feat_name == "full":
            # Use original Phase-1.5 head
            with torch.no_grad():
                probs = torch.sigmoid(h15_original(X.to(DEVICE))).cpu().numpy()
            model_type = "phase1.5"
        else:
            # Use flexible head (untrained)
            flex_head = FlexibleHead(X.shape[1]).to(DEVICE)
            flex_head.eval()
            with torch.no_grad():
                probs = torch.sigmoid(flex_head(X.to(DEVICE))).cpu().numpy()
            model_type = "untrained_flexible"
        
        metrics = compute_metrics(y, probs)
        
        ablation_results["feature_ablation"][feat_name] = {
            "description": feat_desc,
            "feature_dim": int(X.shape[1]),
            "model_type": model_type,
            "metrics": metrics,
            "delta_vs_baseline": {
                k: metrics[k] - baseline_metrics[k] 
                for k in baseline_metrics
            }
        }
        
        print(f"  Dim={X.shape[1]}, F1={metrics['f1']:.4f} "
              f"(Δ{metrics['f1']-baseline_metrics['f1']:+.4f}), "
              f"Acc={metrics['accuracy']:.4f}")

    # ================================================================
    # 4. SUMMARY TABLE
    # ================================================================
    print("\n" + "=" * 60)
    print("[summary] Ablation Study Results")
    print("=" * 60)
    
    summary_table = []
    
    # Baseline
    summary_table.append({
        "experiment": "Baseline (Phase-1)",
        "config": "sent_only, n=N/A",
        "f1": baseline_metrics["f1"],
        "accuracy": baseline_metrics["accuracy"],
        "precision": baseline_metrics["precision"],
        "recall": baseline_metrics["recall"],
        "roc_auc": baseline_metrics["roc_auc"],
    })
    
    # Context window
    for n in window_sizes:
        m = ablation_results["context_window"][f"n={n}"]["metrics"]
        model_type = ablation_results["context_window"][f"n={n}"]["model_type"]
        summary_table.append({
            "experiment": f"Context n={n}",
            "config": f"full features, {model_type}",
            "f1": m["f1"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "roc_auc": m["roc_auc"],
        })
    
    # Feature ablation
    for feat_name in feature_sets.keys():
        m = ablation_results["feature_ablation"][feat_name]["metrics"]
        model_type = ablation_results["feature_ablation"][feat_name]["model_type"]
        summary_table.append({
            "experiment": f"Features: {feat_name}",
            "config": f"n=2, {model_type}",
            "f1": m["f1"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "roc_auc": m["roc_auc"],
        })
    
    ablation_results["summary_table"] = summary_table
    
    # Print table
    print(f"\n{'Experiment':<30} {'F1':>6} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'ROC':>6}")
    print("-" * 70)
    for row in summary_table:
        print(f"{row['experiment']:<30} "
              f"{row['f1']:>6.3f} "
              f"{row['accuracy']:>6.3f} "
              f"{row['precision']:>6.3f} "
              f"{row['recall']:>6.3f} "
              f"{row['roc_auc']:>6.3f}")

    # ================================================================
    # SAVE RESULTS
    # ================================================================
    with open(out_dir / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
    
    # Save summary table as CSV for easy viewing
    import csv
    with open(out_dir / "summary_table.csv", "w", newline="", encoding="utf-8") as f:
        if summary_table:
            writer = csv.DictWriter(f, fieldnames=summary_table[0].keys())
            writer.writeheader()
            writer.writerows(summary_table)

    print("\n" + "=" * 60)
    print(f"[done] Ablation study complete")
    print(f"[done] Results → {out_dir}")
    print(f"       - ablation_results.json (full details)")
    print(f"       - summary_table.csv (quick view)")
    print("=" * 60)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
req_pipeline — MPNet Phase 1.5 Ablation Trainer
--------------------------------------------
Supports different context window sizes and feature combinations.

Context window: configurable (--n-before, --n-after)
Feature sets:
  - sent_only: E_sent (768 dims)
  - sent_ctx: E_sent + E_ctx (1536 dims)
  - sent_ctx_diff: E_sent + E_ctx + abs_diff (2304 dims)
  - full: E_sent + E_ctx + abs_diff + cos (2305 dims, default)
"""

from __future__ import annotations
import os, sys, json, time, math, argparse, random, hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

# ---------------------------------------------------------------------
# ROOT
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
print(f"[init] project root: {PROJECT_ROOT}")

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
TRAIN_PATH = Path("classifier/outputs/splits/train_40.jsonl")
PHASE1_ROOT = Path("classifier/models/mpnet_phase1")
OUT_ROOT = Path("classifier/models/mpnet_phase1_5_ablation")
CACHE_DIR = Path("classifier/datasets/mpnet/cache_ablation")

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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_jsonl(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def sha1_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def latest_run(root: Path) -> str:
    runs = sorted([d.name for d in root.iterdir() if d.is_dir()])
    if not runs:
        raise RuntimeError(f"No runs in {root}")
    return runs[-1]

def mean_pooling(h, m):
    m = m.unsqueeze(-1).type_as(h)
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)

def gold_label(llm_labels):
    return 1 if "requirement" in llm_labels else 0

def build_context(r, n_before=2, n_after=2):
    """Build context string with configurable window size"""
    before = r.get("context_before", [])[-n_before:] if n_before > 0 else []
    after = r.get("context_after", [])[:n_after] if n_after > 0 else []
    sent = (r.get("sentence") or "").strip()

    parts = []
    if before:
        parts.append(" ".join(b.strip() for b in before if isinstance(b, str)))
    parts.append(sent)
    if after:
        parts.append(" ".join(a.strip() for a in after if isinstance(a, str)))

    return " [CTX] ".join(p for p in parts if p)

# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------
class ContextHead(nn.Module):
    """3-layer head for contextual features"""
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
        return self.net(x).squeeze(-1)

# ---------------------------------------------------------------------
# FEATURES
# ---------------------------------------------------------------------
def build_feature_vector(E_s, E_c, feature_set):
    """
    Build feature vector based on selected feature set.
    
    Args:
        E_s: Sentence embeddings (N, 768)
        E_c: Context embeddings (N, 768)
        feature_set: One of ['sent_only', 'sent_ctx', 'sent_ctx_diff', 'full']
    
    Returns:
        Feature tensor (N, D) where D depends on feature_set
    """
    if feature_set == "sent_only":
        # Just sentence embeddings
        return E_s
    
    elif feature_set == "sent_ctx":
        # Sentence + Context concatenation
        return torch.cat([E_s, E_c], dim=1)
    
    elif feature_set == "sent_ctx_diff":
        # Sentence + Context + Absolute difference
        abs_d = torch.abs(E_s - E_c)
        return torch.cat([E_s, E_c, abs_d], dim=1)
    
    elif feature_set == "full":
        # All features: Sentence + Context + Abs diff + Cosine similarity
        abs_d = torch.abs(E_s - E_c)
        cos = nn.functional.cosine_similarity(E_s, E_c).unsqueeze(1)
        return torch.cat([E_s, E_c, abs_d, cos], dim=1)
    
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")


@torch.no_grad()
def compute_features(rows, model_name, max_length, n_before, n_after, 
                     feature_set, device, batch=64):
    """
    Compute features for all samples.
    
    Returns:
        X: Feature tensor (N, D)
        y: Labels (N,)
    """
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = AutoModel.from_pretrained(model_name).to(device).eval()

    for p in enc.parameters():
        p.requires_grad = False

    def embed(texts):
        out = []
        for i in range(0, len(texts), batch):
            b = texts[i:i+batch]
            t = tok(b, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            t = {k: v.to(device) for k, v in t.items()}
            h = enc(**t).last_hidden_state
            e = mean_pooling(h, t["attention_mask"])
            out.append(nn.functional.normalize(e, dim=1).cpu())
        return torch.cat(out, 0)

    # Extract sentences
    sents = [r["sentence"].strip() for r in rows]
    y = torch.tensor([gold_label(r["llm_labels"]) for r in rows], dtype=torch.float32)
    
    # Embed sentences
    print(f"[embed] encoding {len(sents)} sentences...")
    E_s = embed(sents)
    
    # For sent_only, we don't need context
    if feature_set == "sent_only":
        X = build_feature_vector(E_s, E_s, feature_set)
        return X, y
    
    # For all other feature sets, we need context
    print(f"[embed] encoding context (n_before={n_before}, n_after={n_after})...")
    ctxs = [build_context(r, n_before, n_after) for r in rows]
    E_c = embed(ctxs)
    
    # Build final feature vector
    X = build_feature_vector(E_s, E_c, feature_set)
    
    return X, y


def get_feature_dim(feature_set):
    """Get expected feature dimension for a feature set"""
    dims = {
        "sent_only": 768,
        "sent_ctx": 1536,
        "sent_ctx_diff": 2304,
        "full": 2305,
    }
    return dims[feature_set]

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train MPNet Phase-1.5 with ablation configurations")
    
    # Ablation parameters
    ap.add_argument("--n-before", type=int, default=2, 
                    help="Number of sentences before target (0-5)")
    ap.add_argument("--n-after", type=int, default=2,
                    help="Number of sentences after target (0-5)")
    ap.add_argument("--features", type=str, default="full",
                    choices=["sent_only", "sent_ctx", "sent_ctx_diff", "full"],
                    help="Feature set to use")
    
    # Training parameters
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-train", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # Get base model config from Phase-1
    phase1_run = latest_run(PHASE1_ROOT)
    cfg1 = json.loads((PHASE1_ROOT / phase1_run / "config.json").read_text())
    model_name = cfg1["model"]
    max_length = cfg1["max_length"]

    # Load data
    rows = load_jsonl(TRAIN_PATH)
    rows = [r for r in rows if r.get("sentence") and r.get("llm_labels")]
    print(f"[data] usable samples: {len(rows)}")

    # Split train/dev
    y_bin = np.array([gold_label(r["llm_labels"]) for r in rows])
    idx = np.arange(len(rows))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    train_idx, dev_idx = next(sss.split(idx, y_bin))

    # Generate cache key based on configuration
    cache_key = sha1_str(
        f"{TRAIN_PATH}|{model_name}|{max_length}|"
        f"{args.n_before}|{args.n_after}|{args.features}"
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"features_{cache_key}.pt"

    # Compute or load features
    if cache.exists():
        print(f"[cache] loading from {cache.name}")
        blob = torch.load(cache)
        X, y = blob["X"], blob["y"]
    else:
        print(f"[features] computing features...")
        print(f"  - n_before={args.n_before}, n_after={args.n_after}")
        print(f"  - feature_set={args.features}")
        X, y = compute_features(
            rows, model_name, max_length, 
            args.n_before, args.n_after, args.features,
            device
        )
        torch.save({"X": X, "y": y}, cache)
        print(f"[cache] saved to {cache.name}")
    
    expected_dim = get_feature_dim(args.features)
    if X.shape[1] != expected_dim:
        raise RuntimeError(
            f"Feature dimension mismatch: got {X.shape[1]}, expected {expected_dim} "
            f"for feature_set={args.features}"
        )
    
    print(f"[features] shape={X.shape}, positive={y.sum().item():.0f}/{len(y)}")

    # Split features
    Xtr, ytr = X[train_idx], y[train_idx]
    Xdv, ydv = X[dev_idx], y[dev_idx]

    # Create model
    head = ContextHead(X.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    # Setup output directory
    run_id = now_run_id()
    config_str = f"n{args.n_before}-{args.n_after}_{args.features}"
    out = OUT_ROOT / f"{run_id}_{config_str}"
    ensure_dir(out)

    # Save configuration
    config = {
        "model": model_name,
        "max_length": max_length,
        "n_before": args.n_before,
        "n_after": args.n_after,
        "features": args.features,
        "feature_dim": int(X.shape[1]),
        "seed": args.seed,
        "batch_train": args.batch_train,
        "lr": args.lr,
        "epochs": args.epochs,
        "patience": args.patience,
        "n_train": len(train_idx),
        "n_dev": len(dev_idx),
        "run_id": run_id,
    }
    
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    best = float("inf")
    bad = 0
    history = []

    print("=" * 60)
    print(f"[train] Configuration: {config_str}")
    print(f"[train] Feature dim: {X.shape[1]}")
    print(f"[train] Output: {out.name}")
    print("=" * 60)

    for ep in range(1, args.epochs + 1):
        head.train()
        loader = DataLoader(TensorDataset(Xtr, ytr), args.batch_train, shuffle=True)

        tot = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(head(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item()

        avg_train_loss = tot / len(loader)

        # Validation
        head.eval()
        with torch.no_grad():
            logits = head(Xdv.to(device)).cpu()
            loss_d = loss_fn(logits, ydv).item()
            probs = torch.sigmoid(logits).numpy()
            preds = (probs >= 0.5).astype(int)
            
            # Compute metrics
            p, r, f1, _ = precision_recall_fscore_support(
                ydv.numpy(), preds, average="binary", zero_division=0
            )
            acc = accuracy_score(ydv.numpy(), preds)

        print(f"[ep {ep:02d}] train={avg_train_loss:.4f} dev={loss_d:.4f} "
              f"acc={acc:.3f} f1={f1:.3f}")

        # Save checkpoint
        checkpoint = {
            "epoch": ep,
            "head_state": head.state_dict(),
            "optimizer_state": opt.state_dict(),
            "config": config,
        }
        torch.save(checkpoint, out / "last.pt")

        # Track history
        history.append({
            "epoch": ep,
            "train_loss": avg_train_loss,
            "dev_loss": loss_d,
            "dev_acc": float(acc),
            "dev_f1": float(f1),
            "dev_precision": float(p),
            "dev_recall": float(r),
        })

        # Early stopping
        if loss_d < best:
            best = loss_d
            bad = 0
            torch.save(checkpoint, out / "best.pt")
            print(f"[best] dev_loss={best:.4f} f1={f1:.3f}")
        else:
            bad += 1

        if bad >= args.patience:
            print(f"[stop] early stopping (patience={args.patience})")
            break

    # Save training history
    with open(out / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("=" * 60)
    print(f"[done] Training complete")
    print(f"[done] Best dev loss: {best:.4f}")
    print(f"[done] Output: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
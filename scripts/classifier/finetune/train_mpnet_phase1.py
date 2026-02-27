#!/usr/bin/env python3
"""
req_pipeline — MPNet Phase 1 (Sentence-only) Trainer
------------------------------------------------

Trains a sentence-level requirement classifier on top of a frozen MPNet encoder.

Inputs:
- classifier/datasets/mpnet/train_merged.jsonl

Outputs:
- classifier/models/mpnet_phase1/<run_id>/
    - best.pt
    - last.pt
    - config.json
    - metrics_history.jsonl
    - splits.json
    - temperature.json (optional)

Key features:
- Precomputes MPNet embeddings once (encoder frozen)
- Trains only a small classification head
- Evaluates every epoch
- Early stopping + confidence-collapse safety check
- Saves best checkpoint automatically

Usage:
    python scripts/classifier/dataset_normalization/train_mpnet_phase1.py
"""

from __future__ import annotations
import os, sys, json, time, math, argparse, hashlib, random
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score


# ---------------------------------------------------------------------
# SMART ROOT DETECTION
# ---------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[3]  # scripts/classifier/dataset_normalization/ -> req_pipeline-root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

print(f"[init] project root: {PROJECT_ROOT}")


# ---------------------------------------------------------------------
# CONFIG PATHS
# ---------------------------------------------------------------------
DATA_PATH = Path("classifier/datasets/mpnet/train_merged.jsonl")
OUT_ROOT = Path("classifier/models/mpnet_phase1")
CACHE_DIR = Path("classifier/datasets/mpnet/cache")


# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sha1_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def now_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def load_jsonl(p: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # mean pooling over tokens (masked)
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, T, 1]
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def make_strata(label: int, source: str) -> str:
    # keep splits stable across label+source distributions
    return f"{label}::{source or 'unknown'}"


# ---------------------------------------------------------------------
# MODEL: head only
# ---------------------------------------------------------------------
class RequirementHead(nn.Module):
    def __init__(self, in_dim: int = 768, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # logits shape: [B]


# ---------------------------------------------------------------------
# EMBEDDING CACHE
# ---------------------------------------------------------------------
@torch.no_grad()
def compute_embeddings(
    texts: List[str],
    model_name: str,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 128
) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    encoder.eval()
    encoder.to(device)

    # Freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False

    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = encoder(**enc)
        pooled = mean_pooling(out.last_hidden_state, enc["attention_mask"])  # [B, 768]
        pooled = nn.functional.normalize(pooled, p=2, dim=1)  # normalized embeddings help stability
        all_embeds.append(pooled.cpu())

    return torch.cat(all_embeds, dim=0)  # [N, 768]


def get_cache_key(model_name: str, max_length: int) -> str:
    return sha1_str(f"{model_name}::{max_length}::{DATA_PATH.resolve()}")


# ---------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------
@torch.no_grad()
def eval_epoch(
    head: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: torch.device
) -> Dict[str, Any]:
    head.eval()

    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)
    all_logits = []
    all_y = []

    for xb, yb in loader:
        xb = xb.to(device)
        logits = head(xb)  # [B]
        all_logits.append(logits.cpu())
        all_y.append(yb.cpu())

    logits = torch.cat(all_logits, dim=0)
    y_true = torch.cat(all_y, dim=0).numpy()

    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(int)

    # Metrics
    try:
        auc = roc_auc_score(y_true, probs)
    except Exception:
        auc = float("nan")

    acc = accuracy_score(y_true, preds)

    # BCE loss for reporting
    loss_fn = nn.BCEWithLogitsLoss()
    val_loss = loss_fn(logits, torch.tensor(y_true, dtype=torch.float32)).item()

    # Confidence behavior
    req_mask = (y_true == 1)
    non_mask = (y_true == 0)
    mean_req = float(probs[req_mask].mean()) if req_mask.any() else float("nan")
    mean_non = float(probs[non_mask].mean()) if non_mask.any() else float("nan")
    sep = mean_req - mean_non if (not math.isnan(mean_req) and not math.isnan(mean_non)) else float("nan")

    extreme = float(((probs >= 0.95) | (probs <= 0.05)).mean())  # fraction at extremes

    return {
        "val_loss": safe_float(val_loss),
        "roc_auc": safe_float(auc),
        "accuracy": safe_float(acc),
        "mean_p_req": safe_float(mean_req),
        "mean_p_non": safe_float(mean_non),
        "sep": safe_float(sep),
        "extreme_frac": safe_float(extreme),
    }


def confidence_collapse(metrics: Dict[str, Any], extreme_threshold: float = 0.90) -> bool:
    # If most predictions become extreme, your head is probably overfitting
    ext = metrics.get("extreme_frac", 0.0)
    return (not math.isnan(ext)) and ext >= extreme_threshold


# ---------------------------------------------------------------------
# TEMPERATURE SCALING (optional but useful)
# ---------------------------------------------------------------------
def fit_temperature(logits: torch.Tensor, y: torch.Tensor, device: torch.device) -> float:
    """
    Fits a single temperature scalar T > 0 to calibrate logits: logits / T
    using validation negative log-likelihood.
    """
    T = torch.ones(1, device=device, requires_grad=True)

    loss_fn = nn.BCEWithLogitsLoss()

    logits = logits.to(device)
    y = y.to(device).float()

    optimizer = torch.optim.LBFGS([T], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(logits / T.clamp(min=1e-6), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(T.detach().cpu().item())


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train MPNet Phase 1 sentence-only classifier (head-only).")
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2", help="HF model name for encoder")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-embed", type=int, default=64)
    ap.add_argument("--batch-train", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=128)

    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)

    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--eps", type=float, default=0.002)  # val_loss improvement threshold
    ap.add_argument("--min-epochs", type=int, default=5)

    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--test-size", type=float, default=0.1)

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-temperature", action="store_true", help="Skip temperature scaling")
    args = ap.parse_args()

    if not DATA_PATH.exists():
        print(f"[error] missing dataset: {DATA_PATH}")
        sys.exit(1)

    set_seed(args.seed)
    device = torch.device(args.device)

    run_id = now_run_id()
    out_dir = OUT_ROOT / run_id
    ensure_dir(out_dir)
    ensure_dir(CACHE_DIR)

    config = vars(args)
    config.update({
        "run_id": run_id,
        "data_path": str(DATA_PATH),
        "out_dir": str(out_dir),
    })
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"[load] reading {DATA_PATH}")
    rows = load_jsonl(DATA_PATH)
    print(f"[load] samples: {len(rows)}")

    # Basic validation
    texts = []
    labels = []
    sources = []
    for r in rows:
        t = r.get("text", "")
        if not isinstance(t, str) or not t.strip():
            continue
        lab = int(r.get("label", 0))
        src = r.get("source", "unknown")
        texts.append(t.strip())
        labels.append(lab)
        sources.append(src)

    N = len(texts)
    print(f"[prep] usable samples: {N}")

    # Stratified split by (label + source)
    strata = [make_strata(labels[i], sources[i]) for i in range(N)]
    strata = torch.tensor([hash(s) % (2**31-1) for s in strata], dtype=torch.int64).numpy()

    idx = torch.arange(N).numpy()

    # First split: test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    trainval_idx, test_idx = next(sss1.split(idx, strata))

    # Second split: val from trainval
    val_rel = args.val_size / (1.0 - args.test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=args.seed)
    train_idx, val_idx = next(sss2.split(trainval_idx, strata[trainval_idx]))

    train_idx = trainval_idx[train_idx]
    val_idx = trainval_idx[val_idx]

    splits = {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist()
    }
    (out_dir / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    # Embeddings cache
    cache_key = get_cache_key(args.model, args.max_length)
    cache_path = CACHE_DIR / f"embeddings_{cache_key}.pt"
    cache_meta = CACHE_DIR / f"embeddings_{cache_key}.json"

    if cache_path.exists() and cache_meta.exists():
        print(f"[cache] loading embeddings: {cache_path}")
        embeds = torch.load(cache_path, map_location="cpu")
    else:
        print(f"[embed] computing embeddings with {args.model} on {args.device}")
        embeds = compute_embeddings(
            texts=texts,
            model_name=args.model,
            device=device,
            batch_size=args.batch_embed,
            max_length=args.max_length
        )
        torch.save(embeds, cache_path)
        cache_info = {
            "model": args.model,
            "max_length": args.max_length,
            "n": int(embeds.shape[0]),
            "dim": int(embeds.shape[1]),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        cache_meta.write_text(json.dumps(cache_info, indent=2), encoding="utf-8")
        print(f"[cache] saved embeddings: {cache_path}")

    y = torch.tensor(labels, dtype=torch.float32)

    X_train = embeds[train_idx]
    y_train = y[train_idx]
    X_val = embeds[val_idx]
    y_val = y[val_idx]
    X_test = embeds[test_idx]
    y_test = y[test_idx]

    print(f"[split] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # Model head
    head = RequirementHead(in_dim=embeds.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=args.batch_train,
        shuffle=True
    )

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0

    metrics_path = out_dir / "metrics_history.jsonl"

    print("[train] starting Phase 1 (head-only)")
    t0 = time.time()

    for epoch in range(1, args.max_epochs + 1):
        head.train()
        total_loss = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = head(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        train_loss = total_loss / max(1, n_batches)

        # Validate
        val_metrics = eval_epoch(head, X_val, y_val, batch_size=args.batch_train, device=device)
        val_metrics.update({
            "epoch": epoch,
            "train_loss": safe_float(train_loss),
        })

        # Log
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(val_metrics, ensure_ascii=False) + "\n")

        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['val_loss']:.4f} "
            f"auc={val_metrics['roc_auc']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"sep={val_metrics['sep']:.4f} "
            f"extreme={val_metrics['extreme_frac']:.3f}"
        )

        # Save last checkpoint each epoch (cheap)
        torch.save({
            "epoch": epoch,
            "head_state": head.state_dict(),
            "config": config
        }, out_dir / "last.pt")

        # Safety: stop if collapse and we've done some learning
        if epoch >= args.min_epochs and confidence_collapse(val_metrics, extreme_threshold=0.90):
            print("[stop] confidence collapse detected (too many extreme predictions).")
            break

        # Early stopping based on val_loss
        if val_metrics["val_loss"] < best_val_loss - args.eps:
            best_val_loss = val_metrics["val_loss"]
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "head_state": head.state_dict(),
                "best_val_loss": best_val_loss,
                "config": config
            }, out_dir / "best.pt")
            print(f"[best] new best val_loss={best_val_loss:.4f} @ epoch {epoch}")
        else:
            no_improve += 1

        # Stop if no improvement for patience epochs (but respect min_epochs)
        if epoch >= args.min_epochs and no_improve >= args.patience:
            print(f"[stop] no improvement for {args.patience} epochs.")
            break

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"[done] training time: {elapsed:.1f}s")
    print(f"[done] best epoch: {best_epoch} best val_loss: {best_val_loss:.4f}")
    print(f"[done] run dir: {out_dir}")
    print("=" * 60)

    # Final evaluation on test set using best checkpoint
    best_ckpt = out_dir / "best.pt"
    if best_ckpt.exists():
        ck = torch.load(best_ckpt, map_location=device)
        head.load_state_dict(ck["head_state"])
        head.to(device)

    # Compute logits for calibration + reporting
    head.eval()
    with torch.no_grad():
        val_logits = head(X_val.to(device)).cpu()
        test_metrics = eval_epoch(head, X_test, y_test, batch_size=args.batch_train, device=device)

    print(
        f"[test] val_loss_best={best_val_loss:.4f} "
        f"test_loss={test_metrics['val_loss']:.4f} "
        f"test_auc={test_metrics['roc_auc']:.4f} "
        f"test_acc={test_metrics['accuracy']:.4f} "
        f"test_sep={test_metrics['sep']:.4f} "
        f"test_extreme={test_metrics['extreme_frac']:.3f}"
    )

    # Temperature scaling (optional)
    if not args.no_temperature:
        print("[calib] fitting temperature on validation logits")
        T = fit_temperature(val_logits, y_val, device=device)
        (out_dir / "temperature.json").write_text(json.dumps({"temperature": T}, indent=2), encoding="utf-8")
        print(f"[calib] temperature={T:.4f} saved to temperature.json")


if __name__ == "__main__":
    main()

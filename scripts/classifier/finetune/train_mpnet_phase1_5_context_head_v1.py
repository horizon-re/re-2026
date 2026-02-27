#!/usr/bin/env python3
"""
req_pipeline — MPNet Phase 1.5 (Context Head) Trainer
-------------------------------------------------

Trains a context correction head on top of frozen MPNet encoder outputs.

Inputs:
- classifier/outputs/splits/train_40.jsonl   (contextual training pool ~2.2k)

Uses:
- Phase-1 run checkpoint for:
  - encoder model name
  - max_length
  - (optional) Phase-1 head for logging only

Features:
- E_sent = MPNet(sentence)
- E_ctx  = MPNet(context_before[-k] + sentence + context_after[:k])
- X = [E_sent, E_ctx, |E_sent - E_ctx|, cosine(E_sent, E_ctx)]  -> dim = 768*3+1

Outputs:
- classifier/models/mpnet_phase1_5_context/<run_id>/
    - best.pt
    - last.pt
    - config.json
    - metrics_history.jsonl
    - splits.json
    - feature_cache_meta.json
    - feature_cache.pt

Usage:
    python scripts/classifier/finetune/train_mpnet_phase1_5_context_head.py
    python scripts/classifier/finetune/train_mpnet_phase1_5_context_head.py --phase1-run 20260204_040553
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
# SMART ROOT DETECTION
# ---------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
print(f"[init] project root: {PROJECT_ROOT}")

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
TRAIN_PATH_DEFAULT = Path("classifier/outputs/splits/train_40.jsonl")
PHASE1_ROOT = Path("classifier/models/mpnet_phase1")
OUT_ROOT = Path("classifier/models/mpnet_phase1_5_context")
CACHE_DIR = Path("classifier/datasets/mpnet/cache_context")

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

def load_jsonl(p: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def sha1_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def get_latest_run_id(root: Path) -> str:
    runs = [d.name for d in root.iterdir() if d.is_dir()]
    if not runs:
        raise RuntimeError(f"No runs found under: {root}")
    runs.sort()
    return runs[-1]

def mean_pooling(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def gold_label(llm_labels: List[str]) -> int:
    # binary: requirement vs non_requirement
    return 1 if "requirement" in llm_labels else 0

def build_context_text(r: Dict[str, Any], n_before: int, n_after: int) -> str:
    before = r.get("context_before", [])[-n_before:] if n_before > 0 else []
    after = r.get("context_after", [])[:n_after] if n_after > 0 else []
    sent = (r.get("sentence") or "").strip()

    parts = []
    if before:
        parts.append(" ".join([b.strip() for b in before if isinstance(b, str) and b.strip()]))
    parts.append(sent)
    if after:
        parts.append(" ".join([a.strip() for a in after if isinstance(a, str) and a.strip()]))

    return " [CTX] ".join([p for p in parts if p])


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

# ---------------------------------------------------------------------
# MODEL: Context head only
# ---------------------------------------------------------------------
class ContextHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

# ---------------------------------------------------------------------
# FEATURE CACHE
# ---------------------------------------------------------------------
@torch.no_grad()
def compute_features(
    rows: List[Dict[str, Any]],
    model_name: str,
    max_length: int,
    n_before: int,
    n_after: int,
    batch_embed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      X: [N, 768*3 + 1]
      y: [N] float32
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    def embed_texts(texts: List[str]) -> torch.Tensor:
        all_emb = []
        for i in range(0, len(texts), batch_embed):
            batch = texts[i:i+batch_embed]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = encoder(**enc)
            pooled = mean_pooling(out.last_hidden_state, enc["attention_mask"])
            pooled = nn.functional.normalize(pooled, p=2, dim=1)
            all_emb.append(pooled.cpu())
        return torch.cat(all_emb, dim=0)

    sent_texts = [(r.get("sentence") or "").strip() for r in rows]
    ctx_texts = [build_context_text(r, n_before, n_after) for r in rows]
    y = torch.tensor([gold_label(r.get("llm_labels", [])) for r in rows], dtype=torch.float32)

    E_sent = embed_texts(sent_texts)  # [N, 768]
    E_ctx  = embed_texts(ctx_texts)   # [N, 768]

    # |Δ| and cosine
    abs_diff = torch.abs(E_sent - E_ctx)  # [N, 768]
    cos = nn.functional.cosine_similarity(E_sent, E_ctx, dim=1).unsqueeze(1)  # [N, 1]

    X = torch.cat([E_sent, E_ctx, abs_diff, cos], dim=1)  # [N, 2305]
    return X, y

def cache_key(train_path: Path, model_name: str, max_length: int, n_before: int, n_after: int) -> str:
    return sha1_str(f"{train_path.resolve()}::{model_name}::{max_length}::{n_before}::{n_after}")

# ---------------------------------------------------------------------
# EVAL
# ---------------------------------------------------------------------
@torch.no_grad()
def eval_epoch(head: nn.Module, X: torch.Tensor, y: torch.Tensor, batch_size: int, device: torch.device) -> Dict[str, Any]:
    head.eval()
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)

    logits_all = []
    y_all = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = head(xb).cpu()
        logits_all.append(logits)
        y_all.append(yb.cpu())

    logits = torch.cat(logits_all, dim=0)
    y_true = torch.cat(y_all, dim=0).numpy().astype(int)

    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(int)

    loss_fn = nn.BCEWithLogitsLoss()
    val_loss = loss_fn(logits, torch.tensor(y_true, dtype=torch.float32)).item()

    acc = accuracy_score(y_true, preds)
    p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)

    if len(set(y_true.tolist())) == 2:
        auc = roc_auc_score(y_true, probs)
        pr_auc = average_precision_score(y_true, probs)
    else:
        auc, pr_auc = float("nan"), float("nan")

    req_mask = (y_true == 1)
    non_mask = (y_true == 0)
    mean_req = float(probs[req_mask].mean()) if req_mask.any() else float("nan")
    mean_non = float(probs[non_mask].mean()) if non_mask.any() else float("nan")
    sep = mean_req - mean_non if (not math.isnan(mean_req) and not math.isnan(mean_non)) else float("nan")
    extreme = float(((probs >= 0.95) | (probs <= 0.05)).mean())

    return {
        "loss": safe_float(val_loss),
        "accuracy": safe_float(acc),
        "precision": safe_float(p),
        "recall": safe_float(r),
        "f1": safe_float(f1),
        "roc_auc": safe_float(auc),
        "pr_auc": safe_float(pr_auc),
        "mean_p_req": safe_float(mean_req),
        "mean_p_non": safe_float(mean_non),
        "sep": safe_float(sep),
        "extreme_frac": safe_float(extreme),
        "pos_rate": safe_float(float(req_mask.mean())) if len(y_true) else float("nan"),
    }

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train Phase 1.5 contextual head (frozen encoder).")
    ap.add_argument("--train-path", default=str(TRAIN_PATH_DEFAULT))
    ap.add_argument("--phase1-run", default=None, help="Phase-1 run id. If omitted, uses latest.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--n-before", type=int, default=2)
    ap.add_argument("--n-after", type=int, default=2)

    ap.add_argument("--batch-embed", type=int, default=64)
    ap.add_argument("--batch-train", type=int, default=256)

    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)

    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--eps", type=float, default=0.001)
    ap.add_argument("--min-epochs", type=int, default=5)

    ap.add_argument("--dev-size", type=float, default=0.15, help="dev split taken from train-path pool (stratified)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    train_path = Path(args.train_path)
    if not train_path.exists():
        print(f"[error] missing train data: {train_path}")
        sys.exit(1)

    phase1_run = args.phase1_run or get_latest_run_id(PHASE1_ROOT)
    phase1_dir = PHASE1_ROOT / phase1_run
    if not phase1_dir.exists():
        print(f"[error] missing Phase-1 run: {phase1_dir}")
        sys.exit(1)

    cfg1 = json.loads((phase1_dir / "config.json").read_text(encoding="utf-8"))
    model_name = cfg1["model"]
    max_length = int(cfg1["max_length"])

    run_id = now_run_id()
    out_dir = OUT_ROOT / run_id
    ensure_dir(out_dir)
    ensure_dir(CACHE_DIR)

    config = vars(args)
    config.update({
        "run_id": run_id,
        "phase1_run": phase1_run,
        "model_name": model_name,
        "max_length": max_length,
        "train_path": str(train_path),
    })
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = load_jsonl(train_path)
    print(f"[load] {train_path} samples: {len(rows)}")

    # Basic filter
    usable = []
    for r in rows:
        s = r.get("sentence", "")
        labs = r.get("llm_labels", [])
        if isinstance(s, str) and s.strip() and isinstance(labs, list) and len(labs) > 0:
            usable.append(r)
    rows = usable
    N = len(rows)
    print(f"[prep] usable samples: {N}")

    # Stratified dev split by binary label (cheap + stable)
    y_bin = np.array([gold_label(r.get("llm_labels", [])) for r in rows], dtype=int)
    idx = np.arange(N)

    if args.dev_size > 0.0 and N >= 50:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.dev_size, random_state=args.seed)
        train_idx, dev_idx = next(sss.split(idx, y_bin))
    else:
        train_idx, dev_idx = idx, np.array([], dtype=int)

    splits = {"train": train_idx.tolist(), "dev": dev_idx.tolist()}
    (out_dir / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    # Feature cache
    key = cache_key(train_path, model_name, max_length, args.n_before, args.n_after)
    cache_pt = CACHE_DIR / f"ctx_features_{key}.pt"
    cache_meta = CACHE_DIR / f"ctx_features_{key}.json"

    if cache_pt.exists() and cache_meta.exists():
        print(f"[cache] loading features: {cache_pt}")
        blob = torch.load(cache_pt, map_location="cpu")
        X_all = blob["X"]
        y_all = blob["y"]
    else:
        print(f"[feat] computing contextual features with {model_name} on {args.device}")
        X_all, y_all = compute_features(
            rows=rows,
            model_name=model_name,
            max_length=max_length,
            n_before=args.n_before,
            n_after=args.n_after,
            batch_embed=args.batch_embed,
            device=device,
        )
        torch.save({"X": X_all, "y": y_all}, cache_pt)
        meta = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "train_path": str(train_path),
            "model_name": model_name,
            "max_length": max_length,
            "n_before": args.n_before,
            "n_after": args.n_after,
            "N": int(X_all.shape[0]),
            "dim": int(X_all.shape[1]),
        }
        cache_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[cache] saved features: {cache_pt}")

    X_train = X_all[train_idx]
    y_train = y_all[train_idx]
    X_dev = X_all[dev_idx] if len(dev_idx) else None
    y_dev = y_all[dev_idx] if len(dev_idx) else None

    print(f"[split] train={len(train_idx)} dev={len(dev_idx)}")

    head = ContextHead(in_dim=X_all.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=args.batch_train,
        shuffle=True,
    )

    metrics_path = out_dir / "metrics_history.jsonl"

    best = float("inf")
    best_epoch = 0
    no_improve = 0

    print("[train] Phase 1.5 (context head-only) starting")
    t0 = time.time()

    for epoch in range(1, args.max_epochs + 1):
        head.train()
        total = 0.0
        batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = head(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            total += float(loss.item())
            batches += 1

        train_loss = total / max(1, batches)

        # dev metrics if available, else just train metrics
        if X_dev is not None and len(dev_idx) > 0:
            dev_metrics = eval_epoch(head, X_dev, y_dev, batch_size=args.batch_train, device=device)
            main_loss = dev_metrics["loss"]
        else:
            dev_metrics = {}
            main_loss = train_loss

        record = {
            "epoch": epoch,
        "train_loss": safe_float(train_loss),
        }

        if dev_metrics:
            for k, v in dev_metrics.items():
                record[f"dev_{k}"] = v
    
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if dev_metrics:
            print(f"[epoch {epoch:02d}] train_loss={train_loss:.4f} dev_loss={dev_metrics['loss']:.4f} "
                  f"dev_f1={dev_metrics['f1']:.4f} dev_auc={dev_metrics['roc_auc']:.4f} dev_sep={dev_metrics['sep']:.4f}")
        else:
            print(f"[epoch {epoch:02d}] train_loss={train_loss:.4f}")

        # save last
        torch.save({"epoch": epoch, "head_state": head.state_dict(), "config": config}, out_dir / "last.pt")

        # early stopping on main_loss (dev if exists else train)
        if main_loss < best - args.eps:
            best = main_loss
            best_epoch = epoch
            no_improve = 0
            torch.save({"epoch": epoch, "head_state": head.state_dict(), "best_loss": best, "config": config}, out_dir / "best.pt")
            print(f"[best] best_loss={best:.4f} @ epoch {epoch}")
        else:
            no_improve += 1

        if epoch >= args.min_epochs and no_improve >= args.patience:
            print(f"[stop] no improvement for {args.patience} epochs.")
            break

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"[done] Phase 1.5 training time: {elapsed:.1f}s")
    print(f"[done] best epoch: {best_epoch} best_loss: {best:.4f}")
    print(f"[done] output: {out_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()

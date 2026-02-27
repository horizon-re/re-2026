#!/usr/bin/env python3
"""
Plantric — MPNet Context Classifier (LoRA) Trainer (n=1,2,3)
------------------------------------------------------------

Trains 3 LoRA-finetuned MPNet encoders for contextual requirement classification:
  - context window sizes: n_before=n_after in {1,2,3}
  - labels:
      * positive (1): "requirement" OR "with_context"
      * negative (0): "non_requirement"
    Only rows containing at least one of {"requirement","non_requirement","with_context"} are used.

Model:
  - Base encoder: sentence-transformers/all-mpnet-base-v2
  - LoRA applied to MPNet attention projections: q_proj, k_proj, v_proj
  - Classifier head: small MLP on pooled embedding of (context text)

Outputs:
  classifier/models/mpnet_phase1_5_lora_context/<RUN_ID>/
    n1/ {config.json, best.pt, last.pt, history.json}
    n2/ ...
    n3/ ...
"""

from __future__ import annotations

import os, sys, json, time, argparse, random
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score

# PEFT / LoRA
from peft import LoraConfig, get_peft_model

# ---------------------------------------------------------------------
# ROOT
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
print(f"[init] project root: {PROJECT_ROOT}")

# ---------------------------------------------------------------------
# PATHS / CONSTS
# ---------------------------------------------------------------------
DATA_PATH_DEFAULT = Path("classifier/outputs/splits/holdout_60.jsonl")  # you can point to train_60 later
OUT_ROOT = Path("classifier/models/mpnet_phase1_5_lora_context")

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MAX_LENGTH = 128

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
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def mean_pooling(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    mask = attn_mask.unsqueeze(-1).type_as(last_hidden)
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

def is_valid_row(r: Dict[str, Any]) -> bool:
    labels = set(r.get("llm_labels", []) or [])
    return bool(labels.intersection({"requirement", "non_requirement", "with_context"})) and bool(r.get("sentence"))

def gold_label(r: Dict[str, Any]) -> int:
    labels = set(r.get("llm_labels", []) or [])
    # "with_context" is treated as requirement candidate (positive)
    return 1 if ("requirement" in labels or "with_context" in labels) else 0

def build_context(r: Dict[str, Any], n_before: int, n_after: int) -> str:
    before = (r.get("context_before") or [])[-n_before:] if n_before > 0 else []
    after = (r.get("context_after") or [])[:n_after] if n_after > 0 else []
    sent = (r.get("sentence") or "").strip()

    parts: List[str] = []
    if before:
        parts.append(" ".join(b.strip() for b in before if isinstance(b, str) and b.strip()))
    parts.append(sent)
    if after:
        parts.append(" ".join(a.strip() for a in after if isinstance(a, str) and a.strip()))

    # keep tokenization stable
    return " [CTX] ".join([p for p in parts if p])

def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    preds = (probs >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    acc = accuracy_score(y_true, preds)
    roc = float(roc_auc_score(y_true, probs)) if len(set(y_true.tolist())) > 1 else float("nan")
    pr = float(average_precision_score(y_true, probs))
    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
    }

# ---------------------------------------------------------------------
# MODEL HEAD
# ---------------------------------------------------------------------
class ClassifierHead(nn.Module):
    """
    Simple head on top of pooled MPNet embedding.
    (You can make it deeper later if you want, but this is stable.)
    """
    def __init__(self, in_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

# ---------------------------------------------------------------------
# TRAIN LOOP FOR ONE n
# ---------------------------------------------------------------------
def train_for_n(n: int, rows: List[Dict[str, Any]], args, run_root: Path):
    device = torch.device(args.device)
    out_dir = run_root / f"n{n}"
    ensure_dir(out_dir)

    # Build texts + labels
    texts = [build_context(r, n_before=n, n_after=n) for r in rows]
    y = np.array([gold_label(r) for r in rows], dtype=np.int64)

    # Split train/dev
    idx = np.arange(len(rows))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.dev_ratio, random_state=args.seed)
    tr_idx, dv_idx = next(sss.split(idx, y))

    tr_texts = [texts[i] for i in tr_idx]
    dv_texts = [texts[i] for i in dv_idx]
    ytr = torch.tensor(y[tr_idx], dtype=torch.float32)
    ydv = torch.tensor(y[dv_idx], dtype=torch.float32)

    print(f"[data n={n}] train={len(tr_texts)} dev={len(dv_texts)} pos={int(y.sum())}/{len(y)}")

    # Tokenizer + base encoder
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_encoder = AutoModel.from_pretrained(MODEL_NAME)

    # LoRA config for MPNet: q_proj/k_proj/v_proj (NOT query/key/value)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q", "k", "v"],
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )


    encoder = get_peft_model(base_encoder, lora_cfg)
    encoder.to(device)

    head = ClassifierHead(in_dim=768).to(device)

    # Optimizer: encoder LoRA params + head params
    params = list(p for p in encoder.parameters() if p.requires_grad) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    # Save config
    config = {
        "model": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "n_before": n,
        "n_after": n,
        "label_rule": "positive: requirement|with_context; negative: non_requirement; filtered to {requirement,non_requirement,with_context}",
        "seed": args.seed,
        "dev_ratio": args.dev_ratio,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": ["q", "k", "v"],
        },
        "n_total": len(rows),
        "n_train": int(len(tr_texts)),
        "n_dev": int(len(dv_texts)),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    def embed_batch(batch_texts: List[str]) -> torch.Tensor:
        enc = tok(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = encoder(**enc).last_hidden_state
        pooled = mean_pooling(out, enc["attention_mask"])
        pooled = nn.functional.normalize(pooled, p=2, dim=1)
        return pooled

    # Training
    best_loss = float("inf")
    bad = 0
    history = []

    print("=" * 70)
    print(f"[train] LoRA fine-tuning for context n={n}")
    print("=" * 70)

    for ep in range(1, args.epochs + 1):
        encoder.train()
        head.train()

        # shuffle train
        order = np.random.permutation(len(tr_texts))
        tr_texts_shuf = [tr_texts[i] for i in order]
        ytr_shuf = ytr[order]

        # minibatches
        losses = []
        for i in range(0, len(tr_texts_shuf), args.batch_size):
            bt = tr_texts_shuf[i:i + args.batch_size]
            by = ytr_shuf[i:i + args.batch_size].to(device)

            opt.zero_grad()
            emb = embed_batch(bt)
            logits = head(emb)
            loss = loss_fn(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses)) if losses else float("nan")

        # Eval
        encoder.eval()
        head.eval()
        with torch.no_grad():
            dv_probs_all = []
            for i in range(0, len(dv_texts), args.batch_size):
                bt = dv_texts[i:i + args.batch_size]
                emb = embed_batch(bt)
                logits = head(emb)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                dv_probs_all.append(probs)
            dv_probs = np.concatenate(dv_probs_all, axis=0) if dv_probs_all else np.array([])
            dv_loss = float(loss_fn(torch.tensor(np.log(dv_probs / (1 - dv_probs + 1e-9) + 1e-9)).float(),
                                    ydv[:len(dv_probs)].float()).item()) if len(dv_probs) else float("nan")

            metrics = compute_metrics(y[dv_idx], dv_probs) if len(dv_probs) else {}

        print(f"[ep {ep:02d}] train={train_loss:.4f} "
              f"dev_f1={metrics.get('f1', float('nan')):.3f} "
              f"dev_acc={metrics.get('accuracy', float('nan')):.3f}")

        history.append({
            "epoch": ep,
            "train_loss": train_loss,
            "dev_metrics": metrics,
        })

        # Save last
        torch.save({
            "epoch": ep,
            "encoder_state": encoder.state_dict(),
            "head_state": head.state_dict(),
            "config": config,
        }, out_dir / "last.pt")

        # Early stop by dev F1 or dev loss? We'll use dev F1 (primary), fallback to loss.
        score = metrics.get("f1", None)
        if score is None:
            # fallback: minimize train_loss if no metrics (shouldn't happen)
            score = -train_loss

        # We want to MAXIMIZE F1
        is_best = (score > config.get("_best_f1", -1e9))
        if is_best:
            config["_best_f1"] = score
            best_loss = min(best_loss, train_loss)
            bad = 0
            torch.save({
                "epoch": ep,
                "encoder_state": encoder.state_dict(),
                "head_state": head.state_dict(),
                "config": config,
                "best_dev_f1": score,
                "best_dev_metrics": metrics,
            }, out_dir / "best.pt")
            print(f"[best] dev_f1={score:.3f}")
        else:
            bad += 1

        if bad >= args.patience:
            print(f"[stop] early stopping (patience={args.patience})")
            break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[done n={n}] saved → {out_dir}")

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="LoRA fine-tune MPNet for contextual requirement classification (n=1,2,3)")

    ap.add_argument("--data", type=str, default=str(DATA_PATH_DEFAULT), help="JSONL path (filtered by llm_labels)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dev-ratio", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    # LoRA hyperparams
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.1)

    args = ap.parse_args()
    set_seed(args.seed)

    rows_all = load_jsonl(Path(args.data))
    rows = [r for r in rows_all if is_valid_row(r)]
    print(f"[data] total samples: {len(rows)}")

    if len(rows) < 50:
        raise RuntimeError("Too few usable samples after filtering by llm_labels and sentence.")

    run_id = now_run_id()
    run_root = OUT_ROOT / run_id
    ensure_dir(run_root)

    # Save global run metadata
    meta = {
        "run_id": run_id,
        "timestamp": run_id,
        "data_path": args.data,
        "n_total": len(rows),
        "device": args.device,
        "model": MODEL_NAME,
        "notes": "Trains 3 LoRA-finetuned MPNet models for n=1,2,3. Positive = requirement|with_context.",
    }
    (run_root / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for n in [1, 2, 3]:
        print("\n" + "=" * 70)
        train_for_n(n, rows, args, run_root)

    print("\n" + "=" * 70)
    print(f"[done] all runs saved → {run_root}")
    print("=" * 70)

if __name__ == "__main__":
    main()

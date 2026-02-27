#!/usr/bin/env python3
"""
req_pipeline — MPNet Step 2 (Autonomous Encoder Fine-tuning)
-------------------------------------------------------

- Loads Step-1 checkpoint READ-ONLY
- Fine-tunes top-N encoder layers safely
- Evaluates each epoch
- Saves intermediate + best checkpoints
- Stops automatically when learning saturates

Safe to run overnight.
"""

from __future__ import annotations
import os, sys, json, time, math, argparse
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import roc_auc_score, accuracy_score


# ---------------------------------------------------------------------
# ROOT
# ---------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[3]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

print(f"[init] project root: {PROJECT_ROOT}")

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
DATA_PATH = Path("classifier/datasets/mpnet/train_merged.jsonl")
PHASE1_ROOT = Path("classifier/models/mpnet_phase1")
OUT_ROOT = Path("classifier/models/mpnet_phase2_encoder")


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------
class RequirementHead(nn.Module):
    def __init__(self, dim=768, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def mean_pool(h, m):
    m = m.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate(encoder, head, tokenizer, texts, labels, device):
    encoder.eval()
    head.eval()

    logits_all = []
    y_all = []

    for i in range(0, len(texts), 32):
        batch = texts[i:i+32]
        y = labels[i:i+32].to(device)

        enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        out = encoder(**enc)
        emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
        logits = head(emb)

        logits_all.append(logits.cpu())
        y_all.append(y.cpu())

    logits = torch.cat(logits_all)
    y_true = torch.cat(y_all).numpy()
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(int)

    loss = nn.BCEWithLogitsLoss()(logits, torch.tensor(y_true, dtype=torch.float32)).item()
    if len(set(y_true)) < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(y_true, probs)

    acc = accuracy_score(y_true, preds)

    mean_req = probs[y_true == 1].mean()
    mean_non = probs[y_true == 0].mean()
    sep = mean_req - mean_non
    extreme = ((probs >= 0.95) | (probs <= 0.05)).mean()

    return {
        "val_loss": float(loss),
        "roc_auc": float(auc),
        "accuracy": float(acc),
        "sep": float(sep),
        "extreme_frac": float(extreme)
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-run", required=True)
    ap.add_argument("--epochs-max", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--lr-enc", type=float, default=5e-6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    # -------------------------------------------------------------
    # Load Step-1
    # -------------------------------------------------------------
    phase1_path = PHASE1_ROOT / args.phase1_run / "best.pt"
    if not phase1_path.exists():
        raise RuntimeError("Phase-1 checkpoint not found")

    ckpt1 = torch.load(phase1_path, map_location="cpu")

    # -------------------------------------------------------------
    # Output dir
    # -------------------------------------------------------------
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "base_phase1_run.txt").write_text(str(phase1_path))
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    # -------------------------------------------------------------
    # Data
    # -------------------------------------------------------------
    rows = [
        json.loads(l)
        for l in DATA_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
        if l.strip()    
    ]

    texts = [r["text"] for r in rows]
    labels = torch.tensor([int(r["label"]) for r in rows], dtype=torch.float32)

    # simple split reuse
    n = len(texts)
    val_idx = int(n * 0.1)
    texts_val, labels_val = texts[:val_idx], labels[:val_idx]
    texts_train, labels_train = texts[val_idx:], labels[val_idx:]

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    encoder = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").to(device)
    head = RequirementHead().to(device)

    head.load_state_dict(ckpt1["head_state"])

    # freeze all
    for p in encoder.parameters():
        p.requires_grad = False

    # unfreeze top 2 layers
    for layer in encoder.encoder.layer[-2:]:
        for p in layer.parameters():
            p.requires_grad = True

    optimizer = torch.optim.AdamW(
        [
            {"params": head.parameters(), "lr": args.lr_head},
            {"params": [p for p in encoder.parameters() if p.requires_grad], "lr": args.lr_enc}
        ]
    )

    loss_fn = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    no_improve = 0

    metrics_path = out_dir / "metrics_history.jsonl"

    # -------------------------------------------------------------
    # TRAIN LOOP
    # -------------------------------------------------------------
    for epoch in range(1, args.epochs_max + 1):
        encoder.train()
        head.train()

        total_loss = 0.0
        for i in range(0, len(texts_train), 32):
            batch = texts_train[i:i+32]
            y = labels_train[i:i+32].to(device)

            enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

            optimizer.zero_grad()
            out = encoder(**enc)
            emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
            logits = head(emb)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        metrics = evaluate(encoder, head, tokenizer, texts_val, labels_val, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = total_loss

        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        print(f"[epoch {epoch}] val_loss={metrics['val_loss']:.4f} "
              f"auc={metrics['roc_auc']:.4f} sep={metrics['sep']:.4f} "
              f"extreme={metrics['extreme_frac']:.3f}")

        # save last
        torch.save({"encoder": encoder.state_dict(), "head": head.state_dict()}, out_dir / "last.pt")

        # best?
        if metrics["val_loss"] < best_loss - 1e-3:
            best_loss = metrics["val_loss"]
            no_improve = 0
            torch.save({"encoder": encoder.state_dict(), "head": head.state_dict()}, out_dir / "best.pt")
        else:
            no_improve += 1

        # stop conditions
        if no_improve >= args.patience:
            print("[stop] no improvement")
            break

        if metrics["extreme_frac"] > 0.85:
            print("[stop] confidence collapse detected")
            break

    print("=" * 60)
    print(f"[done] Step-2 autonomous training finished")
    print(f"[done] best_val_loss={best_loss:.4f}")
    print(f"[done] output: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

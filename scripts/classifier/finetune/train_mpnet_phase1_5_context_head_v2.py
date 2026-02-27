#!/usr/bin/env python3
"""
req_pipeline — MPNet Phase 1.5 (Context Head) Trainer
-------------------------------------------------
Context window fixed to:
  - 2 sentences before
  - 2 sentences after

Feature dim: 2305
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
OUT_ROOT = Path("classifier/models/mpnet_phase1_5_context_v2")
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
    before = r.get("context_before", [])[-n_before:]
    after = r.get("context_after", [])[:n_after]
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
@torch.no_grad()
def compute_features(rows, model_name, max_length, device, batch=64):
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

    sents = [r["sentence"].strip() for r in rows]
    ctxs = [build_context(r) for r in rows]
    y = torch.tensor([gold_label(r["llm_labels"]) for r in rows], dtype=torch.float32)

    E_s = embed(sents)
    E_c = embed(ctxs)

    abs_d = torch.abs(E_s - E_c)
    cos = nn.functional.cosine_similarity(E_s, E_c).unsqueeze(1)

    X = torch.cat([E_s, E_c, abs_d, cos], dim=1)  # 2305
    return X, y

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-train", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    phase1_run = latest_run(PHASE1_ROOT)
    cfg1 = json.loads((PHASE1_ROOT / phase1_run / "config.json").read_text())
    model_name = cfg1["model"]
    max_length = cfg1["max_length"]

    rows = load_jsonl(TRAIN_PATH)
    rows = [r for r in rows if r.get("sentence") and r.get("llm_labels")]
    print(f"[data] usable samples: {len(rows)}")

    y_bin = np.array([gold_label(r["llm_labels"]) for r in rows])
    idx = np.arange(len(rows))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    train_idx, dev_idx = next(sss.split(idx, y_bin))

    key = sha1_str(f"{TRAIN_PATH}|{model_name}|{max_length}|2|2")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"ctx_features_{key}.pt"

    if cache.exists():
        blob = torch.load(cache)
        X, y = blob["X"], blob["y"]
    else:
        X, y = compute_features(rows, model_name, max_length, device)
        torch.save({"X": X, "y": y}, cache)

    Xtr, ytr = X[train_idx], y[train_idx]
    Xdv, ydv = X[dev_idx], y[dev_idx]

    head = ContextHead(X.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    run_id = now_run_id()
    out = OUT_ROOT / run_id
    ensure_dir(out)

    best = float("inf")
    bad = 0

    print("[train] Phase 1.5 starting")

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

        head.eval()
        with torch.no_grad():
            logits = head(Xdv.to(device)).cpu()
            loss_d = loss_fn(logits, ydv).item()

        print(f"[ep {ep:02d}] train={tot/len(loader):.4f} dev={loss_d:.4f}")

        torch.save({"epoch": ep, "head_state": head.state_dict()}, out / "last.pt")

        if loss_d < best:
            best = loss_d
            bad = 0
            torch.save({"epoch": ep, "head_state": head.state_dict()}, out / "best.pt")
            print(f"[best] {best:.4f}")
        else:
            bad += 1

        if bad >= args.patience:
            print("[stop] early stopping")
            break

    print(f"[done] output → {out}")

if __name__ == "__main__":
    main()

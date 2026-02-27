#!/usr/bin/env python3
"""
Plantric — MPNet FR/NFR Phase 1 Trainer (with Ablations)
-------------------------------------------------------
This is the FIRST-stage classifier for Functional vs Non-Functional Requirements.

- No dependency on any prior phase
- All experiment configuration declared INSIDE the script
- Multiple ablation runs executed sequentially
- Fully reproducible, cache-backed, Plantric-style runs
"""

from __future__ import annotations
import os, sys, json, time, random, hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
TRAIN_PATH = Path("classifier/outputs/splits_frnfr/train.jsonl")
OUT_ROOT = Path("classifier/models/mpnet_frnfr_phase1")
CACHE_DIR = Path("classifier/datasets/mpnet/cache_frnfr")

# ---------------------------------------------------------------------
# GLOBAL EXPERIMENT CONFIG (FROZEN)
# ---------------------------------------------------------------------
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MAX_LENGTH = 256

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_EMBED = 64
BATCH_TRAIN = 256
LR = 1e-4
EPOCHS = 30
PATIENCE = 5
DEV_RATIO = 0.15
THRESH = 0.5

# ---------------------------------------------------------------------
# ABLATION SWEEP (THIS *IS* THE STUDY)
# ---------------------------------------------------------------------
ABLATION_SWEEP: List[Tuple[int, int, str]] = [
    # n_before, n_after, feature_set
    (0, 0, "sent_only"),
    (0, 0, "full"),
    (1, 1, "full"),
    (2, 2, "full"),
    (3, 3, "full"),
    (2, 2, "sent_ctx"),
    (2, 2, "sent_ctx_diff"),
]

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

def sha1_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def load_jsonl(p: Path) -> List[Dict[str, Any]]:
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def mean_pooling(h, m):
    m = m.unsqueeze(-1).type_as(h)
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)

# ---------------------------------------------------------------------
# LABELING
# ---------------------------------------------------------------------
def gold_label_frnfr(label: str) -> int:
    # Positive class = Functional
    return 1 if label.strip().upper() == "F" else 0

# ---------------------------------------------------------------------
# CONTEXT
# ---------------------------------------------------------------------
def safe_list(x):
    return x if isinstance(x, list) else []

def build_context(r: Dict[str, Any], n_before: int, n_after: int) -> str:
    sent = (r.get("sentence") or r.get("text") or "").strip()

    before = safe_list(r.get("context_before"))
    after = safe_list(r.get("context_after"))

    before = before[-n_before:] if n_before > 0 else []
    after = after[:n_after] if n_after > 0 else []

    parts = []
    if before:
        parts.append(" ".join(b.strip() for b in before if isinstance(b, str)))
    parts.append(sent)
    if after:
        parts.append(" ".join(a.strip() for a in after if isinstance(a, str)))

    return " [CTX] ".join(p for p in parts if p)

# ---------------------------------------------------------------------
# MODEL HEAD
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
def feature_dim(feature_set: str) -> int:
    return {
        "sent_only": 768,
        "sent_ctx": 1536,
        "sent_ctx_diff": 2304,
        "full": 2305,
    }[feature_set]

def build_features(E_s, E_c, feature_set: str):
    if feature_set == "sent_only":
        return E_s
    if feature_set == "sent_ctx":
        return torch.cat([E_s, E_c], dim=1)
    if feature_set == "sent_ctx_diff":
        return torch.cat([E_s, E_c, torch.abs(E_s - E_c)], dim=1)
    if feature_set == "full":
        cos = nn.functional.cosine_similarity(E_s, E_c).unsqueeze(1)
        return torch.cat([E_s, E_c, torch.abs(E_s - E_c), cos], dim=1)
    raise ValueError(feature_set)

# ---------------------------------------------------------------------
# EMBEDDING + CACHE
# ---------------------------------------------------------------------
@torch.no_grad()
def compute_features(rows, n_before, n_after, feature_set):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    enc = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    def embed(texts):
        out = []
        for i in range(0, len(texts), BATCH_EMBED):
            b = texts[i:i+BATCH_EMBED]
            t = tok(b, padding=True, truncation=True,
                    max_length=MAX_LENGTH, return_tensors="pt")
            t = {k: v.to(DEVICE) for k, v in t.items()}
            h = enc(**t).last_hidden_state
            e = mean_pooling(h, t["attention_mask"])
            out.append(nn.functional.normalize(e, dim=1).cpu())
        return torch.cat(out, 0)

    texts = []
    labels = []
    for r in rows:
        txt = (r.get("sentence") or r.get("text") or "").strip()
        if not txt:
            continue
        texts.append(txt)
        labels.append(gold_label_frnfr(r["label"]))

    y = torch.tensor(labels, dtype=torch.float32)

    print(f"[embed] encoding {len(texts)} sentences")
    E_s = embed(texts)

    if feature_set == "sent_only":
        return build_features(E_s, E_s, feature_set), y

    ctxs = [build_context(r, n_before, n_after) for r in rows if (r.get("sentence") or r.get("text"))]
    print(f"[embed] encoding context (n={n_before},{n_after})")
    E_c = embed(ctxs)

    return build_features(E_s, E_c, feature_set), y

# ---------------------------------------------------------------------
# TRAIN ONE CONFIG
# ---------------------------------------------------------------------
def train_one(rows, n_before, n_after, feature_set):
    set_seed(SEED)

    cache_key = sha1_str(f"{TRAIN_PATH}|{MODEL_NAME}|{MAX_LENGTH}|{n_before}|{n_after}|{feature_set}")
    ensure_dir(CACHE_DIR)
    cache_path = CACHE_DIR / f"features_{cache_key}.pt"

    if cache_path.exists():
        blob = torch.load(cache_path)
        X, y = blob["X"], blob["y"]
        print(f"[cache] loaded {cache_path.name}")
    else:
        X, y = compute_features(rows, n_before, n_after, feature_set)
        torch.save({"X": X, "y": y}, cache_path)
        print(f"[cache] saved {cache_path.name}")

    assert X.shape[1] == feature_dim(feature_set)

    idx = np.arange(len(y))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=DEV_RATIO, random_state=SEED)
    tr_idx, dv_idx = next(sss.split(idx, y.numpy()))

    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xdv, ydv = X[dv_idx], y[dv_idx]

    head = ContextHead(X.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    run_id = now_run_id()
    out = OUT_ROOT / f"{run_id}_n{n_before}-{n_after}_{feature_set}"
    ensure_dir(out)

    config = {
        "task": "fr_nfr",
        "model": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "n_before": n_before,
        "n_after": n_after,
        "feature_set": feature_set,
        "feature_dim": X.shape[1],
        "seed": SEED,
        "epochs": EPOCHS,
        "patience": PATIENCE,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    best = float("inf")
    bad = 0
    history = []

    for ep in range(1, EPOCHS + 1):
        head.train()
        loader = DataLoader(TensorDataset(Xtr, ytr),
                            batch_size=BATCH_TRAIN, shuffle=True)

        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(head(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item()

        train_loss = tot / len(loader)

        head.eval()
        with torch.no_grad():
            logits = head(Xdv.to(DEVICE)).cpu()
            dev_loss = loss_fn(logits, ydv).item()
            probs = torch.sigmoid(logits).numpy()
            preds = (probs >= THRESH).astype(int)

            p, r, f1, _ = precision_recall_fscore_support(
                ydv.numpy(), preds, average="binary", zero_division=0
            )
            acc = accuracy_score(ydv.numpy(), preds)

        print(f"[ep {ep:02d}] train={train_loss:.4f} dev={dev_loss:.4f} acc={acc:.3f} f1={f1:.3f}")

        history.append({
            "epoch": ep,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "dev_acc": acc,
            "dev_f1": f1,
        })

        ckpt = {"epoch": ep, "state": head.state_dict()}
        torch.save(ckpt, out / "last.pt")

        if dev_loss < best:
            best = dev_loss
            bad = 0
            torch.save(ckpt, out / "best.pt")
        else:
            bad += 1

        if bad >= PATIENCE:
            print("[stop] early stopping")
            break

    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[done] best_dev_loss={best:.4f} -> {out}")

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    set_seed(SEED)
    ensure_dir(OUT_ROOT)

    rows = load_jsonl(TRAIN_PATH)
    rows = [r for r in rows if (r.get("sentence") or r.get("text")) and r.get("label")]

    print(f"[data] usable samples: {len(rows)}")
    print(f"[runs] total ablations: {len(ABLATION_SWEEP)}")

    for i, (nb, na, feat) in enumerate(ABLATION_SWEEP, 1):
        print("=" * 70)
        print(f"[run {i}/{len(ABLATION_SWEEP)}] n_before={nb}, n_after={na}, features={feat}")
        print("=" * 70)
        train_one(rows, nb, na, feat)

    print("[all done] FR/NFR Phase-1 training complete")

if __name__ == "__main__":
    main()

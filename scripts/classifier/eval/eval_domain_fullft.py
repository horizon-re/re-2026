#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Automatic Domain-Only Evaluation (All Models)
---------------------------------------------

Scans:
    classifier/models/context_comparison/

Finds every directory containing:
    - best.pt
    - config.json

Evaluates all models on:
    classifier/outputs/splits/test_rest.jsonl

Outputs:
    classifier/outputs/eval/domain_auto_<timestamp>/
        - summary.json
        - comparison.csv
        - paper_table.tex
        - per-model predictions
"""

from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from transformers import AutoTokenizer, AutoModel

# ---------------------------------------------------------------------
# ROOT
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_ROOT = Path("classifier/models/context_comparison")
TEST_PATH = Path("classifier/outputs/splits/test_rest.jsonl")
OUT_ROOT = Path("classifier/outputs/eval")


# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean_pooling(last_hidden, attn_mask):
    mask = attn_mask.unsqueeze(-1).type_as(last_hidden)
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def build_context_text(row, n_context):
    before = (row.get("context_before") or [])[-n_context:]
    after = (row.get("context_after") or [])[:n_context]
    sent = (row.get("sentence") or "").strip()

    parts = []
    if before:
        parts.append(" ".join(b.strip() for b in before if isinstance(b, str)))
    parts.append(sent)
    if after:
        parts.append(" ".join(a.strip() for a in after if isinstance(a, str)))

    return " [CTX] ".join(parts)


def compute_metrics(y_true, y_pred, probs):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    try:
        roc = roc_auc_score(y_true, probs)
    except:
        roc = float("nan")

    try:
        pr = average_precision_score(y_true, probs)
    except:
        pr = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------
class RequirementClassifier(nn.Module):
    def __init__(self, model_name, config_obj):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.model_name = model_name.lower()

        if hasattr(self.encoder.config, "hidden_size"):
            emb_dim = self.encoder.config.hidden_size
        else:
            emb_dim = 768

        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, config_obj["HIDDEN_DIM_1"]),
            nn.LayerNorm(config_obj["HIDDEN_DIM_1"]),
            nn.ReLU(),
            nn.Dropout(config_obj["DROPOUT_1"]),
            nn.Linear(config_obj["HIDDEN_DIM_1"], config_obj["HIDDEN_DIM_2"]),
            nn.LayerNorm(config_obj["HIDDEN_DIM_2"]),
            nn.ReLU(),
            nn.Dropout(config_obj["DROPOUT_2"]),
            nn.Linear(config_obj["HIDDEN_DIM_2"], 2),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        if "deberta" in self.model_name:
            pooled = outputs.last_hidden_state[:, 0]
        else:
            pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
            pooled = nn.functional.normalize(pooled, p=2, dim=1)

        return self.classifier(pooled)


# ---------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------
def evaluate_model(model_dir, test_rows, y_true, out_dir):

    with open(model_dir / "config.json", "r") as f:
        config = json.load(f)

    model_name = config["model_name"]
    n_context = config["n_context"]
    max_length = config.get("MAX_LENGTH", 256)

    print(f"\n[eval] {model_dir.name} | model={model_name} | k={n_context}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    safe_config = {
        "HIDDEN_DIM_1": config.get("HIDDEN_DIM_1", 512),
        "HIDDEN_DIM_2": config.get("HIDDEN_DIM_2", 256),
        "DROPOUT_1": config.get("DROPOUT_1", 0.3),
        "DROPOUT_2": config.get("DROPOUT_2", 0.2),
    }
    model = RequirementClassifier(model_name, safe_config).to(DEVICE)

    checkpoint = torch.load(model_dir / "best.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ctx_texts = [build_context_text(r, n_context) for r in test_rows]

    all_probs = []
    all_preds = []

    for i in range(0, len(ctx_texts), 32):
        enc = tokenizer(
            ctx_texts[i:i+32],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(enc["input_ids"], enc["attention_mask"])

        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = (probs >= 0.5).astype(int)

        all_probs.extend(probs)
        all_preds.extend(preds)

    metrics = compute_metrics(y_true, np.array(all_preds), np.array(all_probs))

    return {
        "run_dir": model_dir.name,
        "model_name": model_name,
        "n_context": n_context,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / f"domain_auto_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[scan] Scanning models in {MODEL_ROOT}")

    model_dirs = [
        d for d in MODEL_ROOT.iterdir()
        if d.is_dir() and (d / "best.pt").exists()
    ]

    if not model_dirs:
        print("No models found.")
        return

    print(f"[found] {len(model_dirs)} models")

    # Load test data
    rows = load_jsonl(TEST_PATH)

    y_true = []
    filtered = []

    for r in rows:
        labels = set(r.get("llm_labels", []))
        if "requirement" in labels or "with_context" in labels:
            y_true.append(1)
            filtered.append(r)
        elif "non_requirement" in labels:
            y_true.append(0)
            filtered.append(r)

    rows = filtered
    y_true = np.array(y_true)

    print(f"[data] {len(rows)} samples | Pos={y_true.sum()} | Neg={len(y_true)-y_true.sum()}")

    results = []

    for model_dir in model_dirs:
        result = evaluate_model(model_dir, rows, y_true, out_dir)
        results.append(result)

    # Save JSON
    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # CSV
    import csv
    with open(out_dir / "comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run_dir", "model_name", "n_context",
            "accuracy", "precision", "recall",
            "f1", "roc_auc", "pr_auc"
        ])
        writer.writeheader()

        for r in results:
            m = r["metrics"]
            writer.writerow({
                "run_dir": r["run_dir"],
                "model_name": r["model_name"],
                "n_context": r["n_context"],
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
            })

    # LaTeX table
    with open(out_dir / "paper_table.tex", "w") as f:
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("Model & $k$ & Acc & Prec & Rec & F1 \\\\\n")
        f.write("\\midrule\n")
        for r in sorted(results, key=lambda x: (x["model_name"], x["n_context"])):
            m = r["metrics"]
            short_name = "DeBERTa" if "deberta" in r["model_name"].lower() else "MPNet"
            f.write(f"{short_name} & {r['n_context']} & "
                    f"{m['accuracy']:.3f} & "
                    f"{m['precision']:.3f} & "
                    f"{m['recall']:.3f} & "
                    f"{m['f1']:.3f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    print(f"\n[done] Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
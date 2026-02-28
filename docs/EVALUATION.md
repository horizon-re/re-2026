# Evaluation Guide

This guide documents the training methodology used in the study. Full replication may require GPUs, local models, external APIs, and environment configurations beyond the scope of anonymous review. The documentation is provided for transparency rather than out-of-the-box execution.

Complete guide for evaluating context-aware requirements identification models.

## Table of Contents

- [Overview](#overview)
- [Metrics Definitions](#metrics-definitions)
- [Evaluation Scripts](#evaluation-scripts)
- [DeBERTa Comprehensive Evaluation](#deberta-comprehensive-evaluation)
- [MPNet Ablation Study](#mpnet-ablation-study)
- [Domain-Only Evaluation](#domain-only-evaluation)
- [Error Analysis](#error-analysis)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Output Formats](#output-formats)
- [Troubleshooting](#troubleshooting)

---

## Overview

The evaluation framework provides comprehensive assessment of requirements identification models across multiple dimensions:

| Evaluation Script | Purpose | Key Output |
|-------------------|---------|------------|
| `eval_deberta_comp.py` | Multi-context DeBERTa evaluation | summary.json, predictions, errors, CSV |
| `eval_mpnet_phase1_5_holdout_v3.py` | Full ablation study (Phase-1 vs 1.5) | ablation_results.json, summary_table.csv |
| `eval_domain_fullft.py` | Auto-scan all domain models | summary.json, comparison.csv, paper_table.tex |
| `eval_mpnet_phase1_5_holdout.py` | Earlier ablation version | ablation_results.json |
| `metrics.py` | Shared metrics utilities | (currently empty) |

### Label Mapping

All evaluation scripts use the same binary label mapping:
```python
Positive (1): "requirement" OR "with_context" in llm_labels
Negative (0): "non_requirement" in llm_labels
Excluded:     "ambiguous", "non_ambiguous"
```

### Test Sets

- **Track B holdout**: `classifier/outputs/splits/holdout_60.jsonl` (~2,200 instances)
- **Track A test**: `classifier/outputs/splits/test_rest.jsonl` (~600 instances)
- **Track B train (for DeBERTa)**: `classifier/outputs/splits/train_40.jsonl`

---

## Metrics Definitions

All evaluation scripts report the following metrics:

### Primary Metric: F1-Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
Harmonic mean of precision and recall. This is the primary metric used for model selection and comparison throughout the paper.

### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
Overall fraction of correct predictions. Can be misleading with imbalanced classes.

### Precision
```
Precision = TP / (TP + FP)
```
Fraction of predicted requirements that are actually requirements. High precision means few false positives.

### Recall
```
Recall = TP / (TP + FN)
```
Fraction of actual requirements that were correctly identified. High recall means few missed requirements.

### ROC-AUC
Area under the Receiver Operating Characteristic curve. Measures the model's ability to discriminate between classes across all probability thresholds. Range: [0, 1], where 0.5 = random.

### PR-AUC
Area under the Precision-Recall curve. More informative than ROC-AUC for imbalanced datasets. Focuses on the positive class (requirements).

### Confusion Matrix
```
                  Predicted
                  Pos    Neg
Actual  Pos  [  TP  |  FN  ]
        Neg  [  FP  |  TN  ]
```

All metrics are computed using scikit-learn:
```python
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
```

---

## Evaluation Scripts

### Script Overview

```
scripts/classifier/eval/
├── eval_deberta_comp.py                  # Comprehensive DeBERTa evaluation
├── eval_mpnet_phase1_5_holdout_v3.py     # Full ablation study (v3)
├── eval_mpnet_phase1_5_holdout.py        # Earlier ablation version
├── eval_domain_fullft.py                 # Auto-scan domain models
└── metrics.py                            # Shared metrics (placeholder)
```

All scripts share common patterns:
- Auto-detect project root via `Path(__file__).resolve().parents[3]`
- Auto-detect GPU/CPU via `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Timestamped output directories for reproducibility
- JSON + CSV output for both programmatic and human consumption

---

## DeBERTa Comprehensive Evaluation

**Script**: `scripts/classifier/eval/eval_deberta_comp.py`

### Purpose
Evaluates a trained DeBERTa model across multiple context window sizes (n=1, 2, 3) in a single run. Produces detailed per-sample predictions, error analysis, and a comparison CSV.

### Usage

```bash
# Default: auto-detect latest model, test on train_40.jsonl
python scripts/classifier/eval/eval_deberta_comp.py

# Specify model and test data
python scripts/classifier/eval/eval_deberta_comp.py \
  --model-path classifier/models/deberta_full_finetune/20260215_143022_deberta-v3-base \
  --test-data classifier/outputs/splits/test_rest.jsonl

# Custom context sizes and batch size
python scripts/classifier/eval/eval_deberta_comp.py \
  --context-sizes 1,2,3 \
  --batch-size 16 \
  --limit 500
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--test-data` | str | None | Path to test JSONL file (auto-detects if not set) |
| `--model-path` | str | None | Path to model directory (uses latest if not set) |
| `--batch-size` | int | 32 | Inference batch size |
| `--limit` | int | None | Limit number of test samples |
| `--context-sizes` | str | "1,2,3" | Comma-separated context window sizes to evaluate |

### Model Loading

The script auto-discovers models from `classifier/models/deberta_full_finetune/`:
1. Scans for directories containing `best.pt`
2. Sorts by name (timestamp-based) and picks the latest
3. Loads `config.json` for model architecture parameters
4. Reconstructs `RequirementClassifier` with the saved config

The model architecture uses mean pooling over the last hidden state, followed by L2 normalization, then a 3-layer MLP classifier:
```python
encoder → mean_pooling → L2_normalize → Linear(768→512) → LayerNorm → ReLU → 
Dropout(0.3) → Linear(512→256) → LayerNorm → ReLU → Dropout(0.2) → Linear(256→2)
```

### Context Construction

For each context size n, the script builds:
```
[s_{i-n}, ..., s_{i-1}] [CTX] [target sentence] [CTX] [s_{i+1}, ..., s_{i+n}]
```

Context is extracted from `context_before` and `context_after` fields in the JSONL data. Empty or missing context is handled gracefully.

### Output Structure

```
classifier/outputs/eval/deberta_comprehensive_<timestamp>/
├── summary.json              # Overall metrics for all context sizes
├── predictions_n1.json       # Detailed predictions for n=1
├── predictions_n2.json       # Detailed predictions for n=2
├── predictions_n3.json       # Detailed predictions for n=3
├── errors_n1.json            # Misclassified samples for n=1
├── errors_n2.json            # Misclassified samples for n=2
├── errors_n3.json            # Misclassified samples for n=3
└── comparison.csv            # Side-by-side metrics comparison
```

### Output File Formats

#### summary.json
```json
{
  "metadata": {
    "timestamp": "20260215_150000",
    "test_data": "classifier/outputs/splits/train_40.jsonl",
    "model_path": "classifier/models/deberta_full_finetune/...",
    "n_samples": 2212,
    "n_positive": 1423,
    "n_negative": 789,
    "context_sizes": [1, 2, 3]
  },
  "results": [
    {
      "n_context": 1,
      "metrics": {
        "accuracy": 0.861, "precision": 0.893, "recall": 0.871,
        "f1": 0.882, "roc_auc": 0.929, "pr_auc": 0.942,
        "tp": 1239, "fp": 148, "tn": 641, "fn": 184
      }
    }
  ]
}
```

#### predictions_n{k}.json
```json
{
  "n_context": 2,
  "total_samples": 2212,
  "correct": 1935,
  "incorrect": 277,
  "metrics": { ... },
  "predictions": [
    {
      "sent_id": "req-010::s001",
      "sentence": "The system shall process payments.",
      "context_before": ["Payment processing is critical."],
      "context_after": ["This ensures user satisfaction."],
      "true_label": 1,
      "true_labels": ["requirement"],
      "predicted_label": 1,
      "predicted_class": "requirement",
      "prob_requirement": 0.943,
      "prob_non_requirement": 0.057,
      "confidence": 0.943,
      "correct": true,
      "n_context": 2
    }
  ]
}
```

#### errors_n{k}.json
```json
{
  "n_context": 2,
  "total_errors": 277,
  "false_positives": 148,
  "false_negatives": 129,
  "errors": [ ... ]
}
```

#### comparison.csv
```csv
n_context,accuracy,precision,recall,f1,roc_auc,tp,fp,tn,fn
1,86.10,89.30,87.10,88.20,0.9290,1239,148,641,184
2,87.50,89.50,87.10,89.40,0.9330,1239,131,658,184
3,86.50,88.70,87.00,88.50,0.9230,1238,156,633,185
```

---

## MPNet Ablation Study

**Script**: `scripts/classifier/eval/eval_mpnet_phase1_5_holdout_v3.py`

### Purpose
Comprehensive ablation study comparing Phase-1 (sentence-only) vs Phase-1.5 (structured context features) across two dimensions:
1. Context window size: n=1, 2, 3
2. Feature combinations: `sent_only`, `sent_ctx`, `sent_ctx_diff`, `full`

### Usage

```bash
# Default: evaluate on holdout set
python scripts/classifier/eval/eval_mpnet_phase1_5_holdout_v3.py

# With sample limit
python scripts/classifier/eval/eval_mpnet_phase1_5_holdout_v3.py --limit 500

# Custom batch size
python scripts/classifier/eval/eval_mpnet_phase1_5_holdout_v3.py --batch-size 64
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--limit` | int | None | Limit number of evaluation samples |
| `--batch-size` | int | 32 | Embedding batch size |

### Model Discovery

The script auto-discovers trained models from:
- **Phase-1**: `classifier/models/mpnet_phase1/` (latest run)
- **Phase-1.5**: `classifier/models/mpnet_phase1_5_context_v2/` (latest run)
- **Ablation models**: `classifier/models/mpnet_phase1_5_ablation/` (pattern-matched)

Models are loaded via `AdaptiveHead`, which infers architecture from checkpoint keys:
- 1-layer: `net.0.weight` only
- 2-layer: `net.0.weight` + `net.3.weight`
- 3-layer: `net.0.weight` + `net.3.weight` + `net.6.weight` (Phase-1.5 default)

### Ablation Dimensions

#### 1. Context Window Ablation

Tests n=1, 2, 3 with full features (E_s, E_c, |E_s-E_c|, cos):

```python
# For each window size n:
ctx_texts = [" ".join(before[-n:] + [sentence] + after[:n]) for ...]
E_ctx = embed(ctx_texts)
X = [E_sent || E_ctx || |E_sent - E_ctx| || cos(E_sent, E_ctx)]
```

For n=2, uses the original trained Phase-1.5 head. For other window sizes, looks for trained ablation models matching pattern `n{k}-{k}_full` in the ablation directory.

#### 2. Feature Ablation

Tests four feature combinations using n=2 context:

| Feature Set | Dimensions | Components |
|-------------|-----------|------------|
| `sent_only` | 768 | E_s |
| `sent_ctx` | 1536 | E_s ∥ E_c |
| `sent_ctx_diff` | 2304 | E_s ∥ E_c ∥ \|E_s - E_c\| |
| `full` | 2305 | E_s ∥ E_c ∥ \|E_s - E_c\| ∥ cos(E_s, E_c) |

Feature construction:
```python
def build_features(E_sent, E_ctx, feature_set="full"):
    if feature_set == "sent_only":
        return E_sent                                          # 768-dim
    elif feature_set == "sent_ctx":
        return cat([E_sent, E_ctx])                           # 1536-dim
    elif feature_set == "sent_ctx_diff":
        return cat([E_sent, E_ctx, abs(E_sent - E_ctx)])      # 2304-dim
    elif feature_set == "full":
        return cat([E_sent, E_ctx, abs(E_sent - E_ctx),
                    cosine_similarity(E_sent, E_ctx)])         # 2305-dim
```

### Expected Results

#### Context Window Ablation (Full Features)
| Window | F1 | Accuracy | Δ F1 vs Baseline |
|--------|-----|----------|-------------------|
| k=0 (baseline) | 0.664 | 0.621 | — |
| k=1 | 0.824 | 0.794 | +0.160 |
| k=2 | 0.820 | 0.791 | +0.156 |
| k=3 | ~0.815 | ~0.788 | ~+0.151 |

#### Feature Ablation (n=2)
| Feature Set | Dim | F1 | Accuracy | Δ F1 |
|-------------|-----|-----|----------|------|
| sent_only | 768 | 0.664 | 0.621 | — |
| sent_ctx | 1536 | 0.509 | 0.517 | -0.155 |
| sent_ctx_diff | 2304 | 0.725 | 0.569 | +0.061 |
| full | 2305 | 0.820 | 0.791 | +0.156 |

The critical finding: naïve concatenation (`sent_ctx`) actually degrades performance below the sentence-only baseline. The cosine similarity feature in `full` is the key discriminative signal.

### Output Structure

```
classifier/outputs/eval/ablation_<timestamp>/
├── ablation_results.json     # Complete ablation results
└── summary_table.csv         # Quick-view comparison table
```

### ablation_results.json Format
```json
{
  "metadata": {
    "n_samples": 2212,
    "n_positive": 1423,
    "n_negative": 789,
    "phase1_run": "20260210_120000",
    "phase1_5_run": "20260212_140000",
    "timestamp": "20260215_150000"
  },
  "baseline": {
    "phase1": { "accuracy": 0.621, "precision": 0.634, "recall": 0.712, "f1": 0.664, ... }
  },
  "context_window": {
    "n=1": { "n_before": 1, "n_after": 1, "feature_dim": 2305, "metrics": { ... }, "delta_vs_baseline": { ... } },
    "n=2": { ... },
    "n=3": { ... }
  },
  "feature_ablation": {
    "sent_only": { "description": "...", "feature_dim": 768, "metrics": { ... } },
    "sent_ctx": { ... },
    "sent_ctx_diff": { ... },
    "full": { ... }
  },
  "summary_table": [ ... ]
}
```

### Earlier Version: eval_mpnet_phase1_5_holdout.py

The v1 script (`eval_mpnet_phase1_5_holdout.py`) is functionally similar to v3 but lacks:
- Trained ablation model discovery (always uses untrained flexible heads for n≠2)
- The v3 version searches `classifier/models/mpnet_phase1_5_ablation/` for trained models matching each window size

Both versions produce the same output format.

---

## Domain-Only Evaluation

**Script**: `scripts/classifier/eval/eval_domain_fullft.py`

### Purpose
Automatically scans all trained models in `classifier/models/context_comparison/`, evaluates each on the test set, and generates a LaTeX table suitable for direct inclusion in the paper.

### Usage

```bash
# No arguments needed — fully automatic
python scripts/classifier/eval/eval_domain_fullft.py
```

### How It Works

1. **Model Discovery**: Scans `classifier/models/context_comparison/` for directories containing both `best.pt` and `config.json`
2. **Config Loading**: Reads `model_name` and `n_context` from each model's `config.json`
3. **Evaluation**: Loads test data from `classifier/outputs/splits/test_rest.jsonl`, builds context windows per model config, runs inference
4. **Output**: Generates JSON summary, CSV comparison, and LaTeX table

### Model Config Format

Each model directory must contain a `config.json` with:
```json
{
  "model_name": "microsoft/deberta-v3-base",
  "n_context": 2,
  "MAX_LENGTH": 256,
  "HIDDEN_DIM_1": 512,
  "HIDDEN_DIM_2": 256,
  "DROPOUT_1": 0.3,
  "DROPOUT_2": 0.2
}
```

### Pooling Strategy

The script uses architecture-aware pooling:
- **DeBERTa**: First token (`outputs.last_hidden_state[:, 0]`) — CLS-style
- **Other models (MPNet, RoBERTa)**: Mean pooling with L2 normalization

### Output Structure

```
classifier/outputs/eval/domain_auto_<timestamp>/
├── summary.json        # Full results for all models
├── comparison.csv      # Side-by-side CSV comparison
└── paper_table.tex     # LaTeX table for paper inclusion
```

### paper_table.tex Format

```latex
\begin{tabular}{lccccc}
\toprule
Model & $k$ & Acc & Prec & Rec & F1 \\
\midrule
DeBERTa & 1 & 0.861 & 0.893 & 0.871 & 0.882 \\
DeBERTa & 2 & 0.875 & 0.895 & 0.871 & 0.894 \\
DeBERTa & 3 & 0.865 & 0.887 & 0.870 & 0.885 \\
MPNet & 1 & 0.794 & 0.812 & 0.836 & 0.824 \\
\bottomrule
\end{tabular}
```

Models are sorted by `(model_name, n_context)` for consistent ordering. Model names are shortened to "DeBERTa" or "MPNet" based on the model identifier.

### Expected Results (Track A vs Track B)

| Track | Data Size | Best F1 | Best k | ROC-AUC |
|-------|-----------|---------|--------|---------|
| A (Domain-Only) | ~4K | 0.894 | 2 | 0.933 |
| B (Mixed-Source) | ~15K | 0.883 | 2 | 0.933 |

Key insight: Track A with 4K domain-aligned samples outperforms Track B with 15K mixed samples, demonstrating that distribution alignment matters more than scale.

---

## Error Analysis

### Using Prediction Files

The DeBERTa comprehensive evaluation produces detailed error files (`errors_n{k}.json`) that enable systematic error analysis:

```python
import json

# Load errors
with open("classifier/outputs/eval/deberta_comprehensive_.../errors_n2.json") as f:
    data = json.load(f)

print(f"Total errors: {data['total_errors']}")
print(f"False positives: {data['false_positives']}")
print(f"False negatives: {data['false_negatives']}")

# Analyze false positives (predicted requirement, actually not)
fps = [e for e in data["errors"] if e["predicted_label"] == 1]
for e in fps[:5]:
    print(f"  Sentence: {e['sentence'][:80]}...")
    print(f"  Confidence: {e['confidence']:.3f}")
    print(f"  True labels: {e['true_labels']}")
    print()

# Analyze false negatives (missed requirements)
fns = [e for e in data["errors"] if e["predicted_label"] == 0]
for e in fns[:5]:
    print(f"  Sentence: {e['sentence'][:80]}...")
    print(f"  Confidence: {e['confidence']:.3f}")
    print()
```

### Common Error Patterns

1. **Low-confidence false positives**: Sentences with modal verbs that are not requirements (e.g., "The system may be deployed on cloud infrastructure" — capability description, not requirement)
2. **Context-dependent false negatives**: Sentences that are requirements only when read with surrounding context, but the model's context window doesn't capture enough signal
3. **Domain mismatch**: Sentences from external datasets with different linguistic patterns than the primary corpus

### Precision-Recall Trade-off

Adjust the classification threshold to trade precision for recall:
```python
# Default threshold: 0.5
preds = (probs >= 0.5).astype(int)

# Higher threshold: more precision, less recall
preds_strict = (probs >= 0.7).astype(int)

# Lower threshold: more recall, less precision
preds_lenient = (probs >= 0.3).astype(int)
```

---

## Reproducing Paper Results

### RQ1: Impact of Local Context

**Claim**: Adding one neighboring sentence improves F1 by 16 points.

```bash
# Phase-1 baseline (k=0)
python scripts/classifier/finetune/train_mpnet_phase1_5_context_head_v3.py \
  --features sent_only --epochs 30

# Phase-1.5 with k=1
python scripts/classifier/finetune/train_mpnet_phase1_5_context_head_v3.py \
  --features full --n-before 1 --n-after 1 --epochs 30

# Evaluate both
python scripts/classifier/eval/eval_mpnet_phase1_5_holdout_v3.py
```

**Expected**: k=0 F1=0.664 → k=1 F1=0.824 (+16.0 points)

### RQ2: Feature Structure Matters

**Claim**: Naïve concatenation degrades performance; structured divergence features are critical.

```bash
# Run ablation study (evaluates all feature combinations)
python scripts/classifier/eval/eval_mpnet_phase1_5_holdout_v3.py
```

**Expected**:
| Feature Set | F1 |
|-------------|-----|
| sent_only | 0.664 |
| sent_ctx (naïve concat) | 0.509 |
| sent_ctx_diff | 0.725 |
| full (with cosine) | 0.820 |

### RQ3: Distribution vs Architecture

**Claim**: Frozen encoder on 2.2K domain data beats LoRA on 15K mixed data.

```bash
# Frozen encoder (domain data)
python scripts/classifier/finetune/train_mpnet_phase1_5_context_head_v3.py \
  --features full --n-before 1 --n-after 1

# Compare with LoRA results (13 configurations cluster within 2.3 F1 points)
```

**Expected**: Frozen F1=0.824 vs LoRA best F1=0.740

### RQ4: Domain-Only vs Mixed Training

**Claim**: 4K domain-aligned samples outperform 15K mixed samples.

```bash
# Track A (domain-only)
python scripts/classifier/finetune/finetune_deberta.py --track A --context_size 2 --epochs 15

# Track B (mixed)
python scripts/classifier/finetune/finetune_deberta.py --track B --context_size 2 --epochs 15

# Evaluate both
python scripts/classifier/eval/eval_domain_fullft.py
```

**Expected**: Track A F1=0.894 vs Track B F1=0.883

### Complete Results Table (from Paper Section 4)

#### Frozen Encoder (MPNet, Phase-1.5)
| Context | Acc | Prec | Rec | F1 | ROC-AUC |
|---------|-----|------|-----|-----|---------|
| k=0 | 0.621 | 0.634 | 0.712 | 0.664 | 0.645 |
| k=1 | 0.794 | 0.812 | 0.836 | 0.824 | 0.876 |
| k=2 | 0.791 | 0.808 | 0.832 | 0.820 | 0.872 |

#### Full Fine-Tuning (DeBERTa-v3-base, Track A)
| Context | Acc | Prec | Rec | F1 | ROC-AUC |
|---------|-----|------|-----|-----|---------|
| k=1 | 0.861 | 0.893 | 0.871 | 0.882 | 0.929 |
| k=2 | 0.875 | 0.895 | 0.871 | 0.894 | 0.933 |
| k=3 | 0.865 | 0.887 | 0.870 | 0.885 | 0.923 |

#### Full Fine-Tuning (DeBERTa-v3-base, Track B)
| Context | Acc | Prec | Rec | F1 | ROC-AUC |
|---------|-----|------|-----|-----|---------|
| k=1 | 0.868 | 0.893 | 0.871 | 0.882 | 0.929 |
| k=2 | 0.868 | 0.895 | 0.871 | 0.883 | 0.933 |
| k=3 | 0.863 | 0.887 | 0.870 | 0.879 | 0.923 |

#### LoRA Adaptation (Track B, 15K)
| Model | Rank | F1 | Precision | Recall |
|-------|------|-----|-----------|--------|
| DeBERTa-v3-base | 8 | 0.740 | ~0.60 | ~0.92 |
| DeBERTa-v3-base | 16 | 0.729 | ~0.59 | ~0.91 |
| DeBERTa-v3-base | 32 | 0.724 | ~0.59 | ~0.90 |
| MPNet-base | 8 | 0.726 | ~0.59 | ~0.91 |
| Paraphrase-MPNet | 8 | 0.717 | ~0.58 | ~0.90 |

All 13 LoRA configurations cluster within 2.3 F1 points (0.717–0.740), exhibiting precision collapse (~60%) with high recall (~90%).

---

## Output Formats

### Summary of All Output Files

| Script | File | Format | Contents |
|--------|------|--------|----------|
| eval_deberta_comp.py | summary.json | JSON | Metrics for all context sizes |
| eval_deberta_comp.py | predictions_n{k}.json | JSON | Per-sample predictions with probabilities |
| eval_deberta_comp.py | errors_n{k}.json | JSON | Misclassified samples with analysis |
| eval_deberta_comp.py | comparison.csv | CSV | Side-by-side metrics table |
| eval_mpnet_phase1_5_holdout_v3.py | ablation_results.json | JSON | Full ablation study results |
| eval_mpnet_phase1_5_holdout_v3.py | summary_table.csv | CSV | Quick-view comparison |
| eval_domain_fullft.py | summary.json | JSON | All model results |
| eval_domain_fullft.py | comparison.csv | CSV | Model comparison table |
| eval_domain_fullft.py | paper_table.tex | LaTeX | Publication-ready table |

### Output Directory Convention

All evaluation outputs are saved under `classifier/outputs/eval/` with timestamped subdirectories:
```
classifier/outputs/eval/
├── deberta_comprehensive_20260215_150000/
├── ablation_20260215_160000/
└── domain_auto_20260215_170000/
```

---

## Troubleshooting

### No Models Found

```
[error] No trained models found
```

**Solution**: Train models first (see [TRAINING.md](TRAINING.md)):
```bash
python scripts/classifier/finetune/finetune_deberta.py --track A --context_size 2
```

### No Test Data Found

```
[error] No test data found!
```

**Solution**: Run the dataset preparation pipeline:
```bash
python scripts/requirement-unitizer/merge_and_resplit.py
```

### CUDA Out of Memory During Evaluation

**Solution**: Reduce batch size:
```bash
python scripts/classifier/eval/eval_deberta_comp.py --batch-size 8
```

### Mismatched Model Architecture

```
RuntimeError: Error(s) in loading state_dict
```

**Solution**: Ensure the `config.json` matches the checkpoint. The classifier head dimensions (`HIDDEN_DIM_1`, `HIDDEN_DIM_2`) must match what was used during training.

### NaN in ROC-AUC

This occurs when the test set contains only one class. Ensure your test data has both positive and negative samples:
```python
import json
rows = [json.loads(l) for l in open("test.jsonl")]
labels = [r.get("llm_labels", []) for r in rows]
print(f"Has requirement: {any('requirement' in l for l in labels)}")
print(f"Has non_requirement: {any('non_requirement' in l for l in labels)}")
```

---

## References

- **Training Guide**: [TRAINING.md](TRAINING.md)
- **Dataset Guide**: [DATASET.md](DATASET.md)
- **Scripts Reference**: [SCRIPTS.md](SCRIPTS.md)
- **Pipeline Overview**: [PIPELINE.md](PIPELINE.md)

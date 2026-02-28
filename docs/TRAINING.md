# Model Training Guide

This guide documents the training methodology used in the study. Full replication may require GPUs, local models, external APIs, and environment configurations beyond the scope of anonymous review. The documentation is provided for transparency rather than out-of-the-box execution.

Complete guide for training context-aware requirements identification models.

## Table of Contents

- [Overview](#overview)
- [Training Approaches](#training-approaches)
- [Phase 1: Frozen Encoder Baseline](#phase-1-frozen-encoder-baseline)
- [Phase 1.5: Structured Context Features](#phase-15-structured-context-features)
- [Phase 2: Full Fine-Tuning](#phase-2-full-fine-tuning)
- [LoRA Adaptation](#lora-adaptation)
- [Configuration Guide](#configuration-guide)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Training Monitoring](#training-monitoring)
- [Troubleshooting](#troubleshooting)

---

## Overview

This guide covers three main training approaches, each representing different trade-offs between performance, computational cost, and training time:

| Approach | Parameters Updated | Training Time | Best F1 | Use Case |
|----------|-------------------|---------------|---------|----------|
| Frozen Encoder | ~1M (head only) | ~30 min | 0.824 | Quick baseline, limited GPU |
| LoRA | ~2-8M (adapters) | ~2 hours | 0.740 | Parameter-efficient |
| Full Fine-Tuning | ~184M (all) | ~4 hours | 0.894 | Best performance |

### Key Results Summary

**Frozen Encoder (MPNet)**:
- Context k=0: F1=0.664
- Context k=1: F1=0.824 (+16.0 points)
- Context k=2: F1=0.820

**Full Fine-Tuning (DeBERTa, Track A)**:
- Context k=1: F1=0.882
- Context k=2: F1=0.894 (Best)
- Context k=3: F1=0.885

---

## Training Approaches

### Approach Comparison

```
Frozen Encoder (Phase 1.5)
├── Encoder: Frozen (no updates)
├── Features: Structured (E_s, E_c, |E_s-E_c|, cos)
├── Head: 3-layer MLP
└── Training: Head only

LoRA Adaptation
├── Encoder: Low-rank adapters
├── Rank: 8-32
├── Head: Task-specific
└── Training: Adapters + head

Full Fine-Tuning (Phase 2)
├── Encoder: All parameters
├── Features: End-to-end
├── Head: 3-layer MLP
└── Training: Everything
```

---

## Phase 1: Frozen Encoder Baseline

### Overview
Sentence-only classification with frozen MPNet encoder.

**Performance**: F1=0.664, Acc=0.621

### Quick Start
```bash
# Not explicitly implemented as separate script
# Use Phase 1.5 with --features sent_only
python scripts/classifier/finetune/train_mpnet_phase1_5_context_head_v3.py \
  --features sent_only \
  --epochs 30
```

### Architecture
```python
# Frozen encoder
encoder = AutoModel.from_pretrained("all-mpnet-base-v2")
for param in encoder.parameters():
    param.requires_grad = False

# Simple classifier head
classifier = nn.Sequential(
    nn.Linear(768, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 1)
)
```

### Training Configuration
```python
LEARNING_RATE = 1e-4
BATCH_SIZE = 256
EPOCHS = 30
PATIENCE = 5
WEIGHT_DECAY = 1e-4
```

---

## Phase 1.5: Structured Context Features

### Overview
Frozen encoder with explicit context modeling using structured divergence features.

**Performance**: F1=0.824 (k=1), Acc=0.794

### Key Innovation
Instead of naïve concatenation, uses structured features:
```python
features = [
    E_s,              # Sentence embedding (768-dim)
    E_c,              # Context embedding (768-dim)
    |E_s - E_c|,      # Absolute difference (768-dim)
    cos(E_s, E_c)     # Cosine similarity (1-dim)
]
# Total: 2305-dim
```

### Quick Start
```bash
# Full feature set (default)
python scripts/classifier/finetune/train_mpnet_phase1_5_context_head_v3.py \
  --n-before 1 \
  --n-after 1 \
  --features full \
  --epochs 30

# Ablation: sentence + context only
python scripts/classifier/finetune/train_mpnet_phase1_5_context_head_v3.py \
  --n-before 1 \
  --n-after 1 \
  --features sent_ctx \
  --epochs 30
```


### Feature Sets

#### 1. `sent_only` (768-dim)
Just sentence embeddings, no context.
```python
X = E_s
```

#### 2. `sent_ctx` (1536-dim)
Sentence + context concatenation.
```python
X = [E_s || E_c]
```
**Note**: This degrades performance (F1=0.509) due to lack of divergence signal.

#### 3. `sent_ctx_diff` (2304-dim)
Adds absolute difference.
```python
X = [E_s || E_c || |E_s - E_c|]
```

#### 4. `full` (2305-dim) - **Recommended**
All features including cosine similarity.
```python
X = [E_s || E_c || |E_s - E_c| || cos(E_s, E_c)]
```

### Context Window Configuration

```bash
# k=1 (1 sentence before and after)
--n-before 1 --n-after 1

# k=2 (2 sentences before and after)
--n-before 2 --n-after 2

# k=3 (3 sentences before and after)
--n-before 3 --n-after 3

# Asymmetric (2 before, 1 after)
--n-before 2 --n-after 1
```

### Complete Configuration

**Script**: `scripts/classifier/finetune/train_mpnet_phase1_5_context_head_v3.py`

**Arguments**:
```bash
--n-before INT        # Sentences before target (0-5)
--n-after INT         # Sentences after target (0-5)
--features STR        # Feature set: sent_only|sent_ctx|sent_ctx_diff|full
--seed INT            # Random seed (default: 42)
--batch-train INT     # Batch size (default: 256)
--lr FLOAT            # Learning rate (default: 1e-4)
--epochs INT          # Training epochs (default: 30)
--patience INT        # Early stopping patience (default: 5)
--device STR          # Device: cuda|cpu
```

### Training Process

1. **Feature Extraction**:
   ```python
   # Encode sentences
   E_s = encoder.encode(sentences)
   
   # Build context strings
   context = [s_{i-k}, ..., s_{i-1}] [CTX] s_i [CTX] [s_{i+1}, ..., s_{i+k}]
   
   # Encode context
   E_c = encoder.encode(contexts)
   
   # Compute divergence features
   abs_diff = |E_s - E_c|
   cos_sim = cos(E_s, E_c)
   
   # Concatenate
   X = [E_s || E_c || abs_diff || cos_sim]
   ```

2. **Caching**:
   Features are cached to avoid recomputation:
   ```
   classifier/datasets/mpnet/cache_ablation/features_{hash}.pt
   ```

3. **Training**:
   ```python
   # 3-layer MLP head
   head = nn.Sequential(
       nn.Linear(2305, 512),
       nn.ReLU(),
       nn.Dropout(0.3),
       nn.Linear(512, 128),
       nn.ReLU(),
       nn.Dropout(0.2),
       nn.Linear(128, 1)
   )
   
   # Binary cross-entropy loss
   loss = BCEWithLogitsLoss()
   
   # AdamW optimizer
   optimizer = AdamW(head.parameters(), lr=1e-4, weight_decay=1e-4)
   ```

### Expected Results

| Configuration | F1 | Accuracy | ROC-AUC |
|--------------|-----|----------|---------|
| sent_only (k=0) | 0.664 | 0.621 | 0.645 |
| sent_ctx (k=1) | 0.509 | 0.517 | 0.542 |
| sent_ctx_diff (k=1) | 0.725 | 0.569 | 0.549 |
| full (k=1) | 0.824 | 0.794 | 0.876 |
| full (k=2) | 0.820 | 0.791 | 0.872 |

### Output Structure
```
classifier/models/mpnet_phase1_5_ablation/
└── {timestamp}_n{before}-{after}_{features}/
    ├── config.json          # Training configuration
    ├── best.pt             # Best checkpoint (by dev loss)
    ├── last.pt             # Last checkpoint
    └── history.json        # Training history
```

---

## Phase 2: Full Fine-Tuning

### Overview
End-to-end fine-tuning of all model parameters for maximum performance.

**Performance**: F1=0.894 (Track A, k=2), Acc=0.875

### Quick Start

**Track A (Domain-Only, 4K samples)**:
```bash
python scripts/classifier/finetune/finetune_deberta.py \
  --track A \
  --context_size 2 \
  --epochs 15 \
  --lr 5e-6 \
  --batch_size 8
```

**Track B (Mixed-Source, 15K samples)**:
```bash
python scripts/classifier/finetune/finetune_deberta.py \
  --track B \
  --context_size 2 \
  --epochs 15 \
  --lr 5e-6 \
  --batch_size 8
```

### Model Architecture

**Script**: `scripts/classifier/finetune/finetune_deberta.py`

**Encoder Options**:
```python
# DeBERTa-v3-base (184M params) - Recommended
MODEL_NAME = "microsoft/deberta-v3-base"

# DeBERTa-v3-large (304M params) - Best accuracy, slower
MODEL_NAME = "microsoft/deberta-v3-large"

# RoBERTa-large (355M params) - Alternative
MODEL_NAME = "roberta-large"

# MPNet-base (110M params) - Faster
MODEL_NAME = "microsoft/mpnet-base"
```

**Classifier Head**:
```python
classifier = nn.Sequential(
    nn.Linear(768, 512),
    nn.LayerNorm(512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.LayerNorm(256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 2)  # Binary classification
)
```

### Configuration Class

All hyperparameters in one place:
```python
class Config:
    # Model
    MODEL_NAME = "microsoft/deberta-v3-base"
    FULL_FINETUNE = True
    
    # Training
    EPOCHS = 15
    PATIENCE = 5
    BATCH_SIZE = 8
    ACCUMULATION_STEPS = 4  # Effective batch = 32
    
    LEARNING_RATE = 5e-6
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1.0
    SCHEDULER = "cosine"
    
    # Data
    USE_FRNFR = True
    USE_SYNTHETIC = True
    DEV_RATIO = 0.15
    
    # Context
    N_CONTEXT = 2
    MAX_LENGTH = 256
    
    # Architecture
    HIDDEN_DIM_1 = 512
    HIDDEN_DIM_2 = 256
    DROPOUT_1 = 0.3
    DROPOUT_2 = 0.2
    
    # Class imbalance
    USE_CLASS_WEIGHTS = True
```

### Data Loading

**Track A (Domain-Only)**:
```python
# Load primary corpus only
samples = load_contextual_data("classifier/outputs/splits/train_4k.jsonl", k=2)

# Split: 85% train, 15% dev
train_samples, dev_samples = stratified_split(samples, test_size=0.15)
```

**Track B (Mixed-Source)**:
```python
# Load all sources
contextual = load_contextual_data("classifier/outputs/splits/holdout_60.jsonl", k=2)
frnfr = load_frnfr_data("classifier/outputs/splits_frnfr/train.jsonl")
synthetic = load_synthetic_nr("classifier/synthetic/non_requirements/synthetic_non_requirements.jsonl")

# Combine
all_samples = contextual + frnfr + synthetic

# Balance classes (60-65% positive)
balanced_samples = balance_classes(all_samples, target_ratio=0.625)
```

### Context Construction

```python
def build_context(row, n_context=2):
    """
    Build context window around target sentence.
    
    Example (k=2):
    [s_{i-2}] [s_{i-1}] [CTX] [target] [CTX] [s_{i+1}] [s_{i+2}]
    """
    before = row.get("context_before", [])[-n_context:]
    after = row.get("context_after", [])[:n_context]
    target = row.get("sentence", "").strip()
    
    parts = []
    if before:
        parts.append(" ".join(before))
    parts.append(target)
    if after:
        parts.append(" ".join(after))
    
    return " [CTX] ".join(parts)
```

### Training Loop

```python
def train_epoch(model, train_loader, optimizer, scheduler, criterion, device, config):
    model.train()
    optimizer.zero_grad()
    
    for step, batch in enumerate(train_loader):
        # Forward
        logits = model(batch['input_ids'], batch['attention_mask'])
        loss = criterion(logits, batch['labels'])
        
        # Backward with gradient accumulation
        loss = loss / config.ACCUMULATION_STEPS
        loss.backward()
        
        if (step + 1) % config.ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    
    return metrics
```

### Loss Function

**Class-Weighted Cross-Entropy**:
```python
# Compute class weights
pos_weight = n_negative / n_positive
class_weights = torch.tensor([1.0, pos_weight])

# Loss function
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### Learning Rate Schedule

**Cosine with Warmup** (Recommended):
```python
num_training_steps = len(train_loader) * epochs // accumulation_steps
num_warmup_steps = int(num_training_steps * 0.1)  # 10% warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)
```

**Linear with Warmup** (Alternative):
```python
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)
```


### Early Stopping

```python
best_f1 = -1.0
patience_counter = 0
PATIENCE = 5

for epoch in range(epochs):
    train_metrics = train_epoch(...)
    dev_metrics = evaluate(...)
    
    if dev_metrics['f1'] > best_f1:
        best_f1 = dev_metrics['f1']
        save_checkpoint("best.pt")
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= PATIENCE:
        print("Early stopping")
        break
```

### Expected Results

**Track A (Domain-Only, 4K samples)**:
| Context | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---------|----------|-----------|--------|-----|---------|
| k=1 | 0.861 | 0.893 | 0.871 | 0.882 | 0.929 |
| k=2 | 0.875 | 0.895 | 0.871 | 0.894 | 0.933 |
| k=3 | 0.865 | 0.887 | 0.870 | 0.885 | 0.923 |

**Track B (Mixed-Source, 15K samples)**:
| Context | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---------|----------|-----------|--------|-----|---------|
| k=1 | 0.868 | 0.893 | 0.871 | 0.882 | 0.929 |
| k=2 | 0.868 | 0.895 | 0.871 | 0.883 | 0.933 |
| k=3 | 0.863 | 0.887 | 0.870 | 0.879 | 0.923 |

### Output Structure
```
classifier/models/deberta_full_finetune/
└── {timestamp}_deberta-v3-base/
    ├── config.json          # Full configuration
    ├── best.pt             # Best checkpoint (by F1)
    ├── last.pt             # Last checkpoint
    └── history.json        # Training history
```

### Checkpoint Format
```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "config": config_dict,
    "dev_metrics": {
        "accuracy": 0.875,
        "precision": 0.895,
        "recall": 0.871,
        "f1": 0.894,
        "roc_auc": 0.933,
        "pr_auc": 0.942
    }
}
```

---

## LoRA Adaptation

### Overview
Parameter-efficient fine-tuning using Low-Rank Adaptation.

**Performance**: F1=0.740 (best), Acc=0.628

**Note**: LoRA underperforms frozen encoder baseline due to distribution mismatch in Track B data.

### Configuration

```python
# LoRA parameters
LORA_RANK = 8  # or 16, 32
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# Target modules (DeBERTa)
TARGET_MODULES = [
    "query_proj",
    "key_proj",
    "value_proj",
    "dense"
]
```

### Results Across Configurations

| Model | Rank | LR | Context | F1 |
|-------|------|-----|---------|-----|
| DeBERTa-v3-base | 8 | 1e-5 | k=1 | 0.740 |
| DeBERTa-v3-base | 16 | 1e-5 | k=2 | 0.729 |
| DeBERTa-v3-base | 32 | 1e-5 | k=3 | 0.724 |
| MPNet-base | 8 | 1e-5 | k=1 | 0.726 |
| Paraphrase-MPNet | 8 | 1e-5 | k=1 | 0.717 |

**Key Finding**: All LoRA configurations cluster within 2.3 F1 points (0.717-0.740), indicating distributional bottleneck rather than architectural limitation.

### Precision Collapse Issue

All LoRA models exhibit:
- High recall: ~90-93%
- Low precision: ~59-62%
- Many false positives: ~789 per 2,212 test instances

**Cause**: External datasets (DOSSPRE, HuggingFace) contain only canonical requirements with explicit modal verbs, biasing toward over-prediction.

---

## Configuration Guide

### Choosing an Approach

**Use Frozen Encoder (Phase 1.5) if**:
- Limited GPU memory (<8GB)
- Quick baseline needed
- Interpretable features desired
- Training time < 1 hour

**Use LoRA if**:
- Moderate GPU memory (8-16GB)
- Parameter efficiency important
- Multiple experiments needed
- Training time ~2 hours

**Use Full Fine-Tuning if**:
- Sufficient GPU memory (>16GB)
- Best performance required
- Production deployment planned
- Training time ~4 hours acceptable

### Model Selection

**DeBERTa-v3-base** (Recommended):
- 184M parameters
- Best F1: 0.894
- Training time: ~4 hours
- GPU memory: ~16GB

**DeBERTa-v3-large**:
- 304M parameters
- Slightly better accuracy
- Training time: ~8 hours
- GPU memory: ~24GB

**MPNet-base**:
- 110M parameters
- Faster training
- Training time: ~2 hours
- GPU memory: ~12GB

### Context Window Selection

**k=1** (Recommended for quick experiments):
- Avg context length: 36.6 tokens
- Truncation rate: 2.3%
- F1 improvement: +16.0 points

**k=2** (Recommended for best performance):
- Avg context length: 73.2 tokens
- Truncation rate: 8.7%
- Consistently optimal across architectures

**k=3**:
- Avg context length: 109.8 tokens
- Truncation rate: 18.4%
- Diminishing returns

### Batch Size Guidelines

| GPU Memory | Batch Size | Accumulation | Effective Batch |
|------------|------------|--------------|-----------------|
| 8GB | 4 | 8 | 32 |
| 12GB | 8 | 4 | 32 |
| 16GB | 16 | 2 | 32 |
| 24GB | 32 | 1 | 32 |

**Formula**: `effective_batch = batch_size × accumulation_steps`

---

## Hyperparameter Tuning

### Learning Rate

**Frozen Encoder**:
```python
LR = 1e-4  # Higher LR for head-only training
```

**Full Fine-Tuning**:
```python
LR = 5e-6  # Lower LR to avoid catastrophic forgetting
```

**LoRA**:
```python
LR = 1e-5  # Middle ground
```

### Learning Rate Warmup

```python
WARMUP_RATIO = 0.1  # 10% of training steps

# Example: 15 epochs, 100 steps/epoch, 4 accumulation
total_steps = 15 * 100 // 4 = 375
warmup_steps = int(375 * 0.1) = 37
```

### Weight Decay

```python
# Frozen encoder
WEIGHT_DECAY = 1e-4

# Full fine-tuning
WEIGHT_DECAY = 1e-2  # Stronger regularization
```

### Dropout

```python
# Classifier head
DROPOUT_1 = 0.3  # First layer
DROPOUT_2 = 0.2  # Second layer

# Encoder (if unfrozen)
ATTENTION_DROPOUT = 0.1
HIDDEN_DROPOUT = 0.1
```

### Class Weights

```python
# Compute from training data
n_positive = sum(1 for s in train_samples if s.label == 1)
n_negative = len(train_samples) - n_positive

pos_weight = n_negative / n_positive
class_weights = torch.tensor([1.0, pos_weight])

# Example: 60% positive, 40% negative
# pos_weight = 0.4 / 0.6 = 0.667
# class_weights = [1.0, 0.667]
```

### Grid Search Example

```bash
# Context window ablation
for k in 1 2 3; do
    python scripts/classifier/finetune/finetune_deberta.py \
        --context_size $k \
        --epochs 15
done

# Learning rate search
for lr in 1e-6 5e-6 1e-5; do
    python scripts/classifier/finetune/finetune_deberta.py \
        --lr $lr \
        --epochs 15
done

# Batch size search
for bs in 4 8 16; do
    python scripts/classifier/finetune/finetune_deberta.py \
        --batch_size $bs \
        --epochs 15
done
```

---

## Training Monitoring

### Console Output

```
[EPOCH 5 RESULTS]
======================================================================

TRAINING METRICS:
  Loss:      0.2341
  Accuracy:  89.23%
  Precision: 91.45%
  Recall:    87.12%
  F1-Score:  89.24%
  ROC-AUC:   0.9456

DEV METRICS:
  Loss:      0.2789
  Accuracy:  87.50%
  Precision: 89.50%
  Recall:    87.10%
  F1-Score:  88.28%
  ROC-AUC:   0.9330

CONFUSION MATRIX (Dev):
  True Positives:   523
  False Positives:   63
  True Negatives:   537
  False Negatives:   77

PROGRESS:
  Best F1 so far: 89.40%
  Patience: 1/5
======================================================================
```

### Training History

```json
{
  "epoch": 5,
  "train_loss": 0.2341,
  "train_metrics": {
    "accuracy": 0.8923,
    "precision": 0.9145,
    "recall": 0.8712,
    "f1": 0.8924,
    "roc_auc": 0.9456
  },
  "dev_loss": 0.2789,
  "dev_metrics": {
    "accuracy": 0.8750,
    "precision": 0.8950,
    "recall": 0.8710,
    "f1": 0.8828,
    "roc_auc": 0.9330
  }
}
```

### TensorBoard (Optional)

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(f"runs/{run_id}")

# Log metrics
writer.add_scalar("Loss/train", train_loss, epoch)
writer.add_scalar("Loss/dev", dev_loss, epoch)
writer.add_scalar("F1/train", train_f1, epoch)
writer.add_scalar("F1/dev", dev_f1, epoch)

# Log learning rate
writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

# View in browser
# tensorboard --logdir runs/
```

### Weights & Biases (Optional)

```python
import wandb

wandb.init(project="requirements-classification", config=config_dict)

# Log metrics
wandb.log({
    "epoch": epoch,
    "train_loss": train_loss,
    "dev_loss": dev_loss,
    "dev_f1": dev_f1,
    "dev_accuracy": dev_acc
})

# Log best model
wandb.save("best.pt")
```

---

## Troubleshooting

### Out of Memory (OOM)

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
```bash
# Reduce batch size
--batch_size 4

# Increase gradient accumulation
--accumulation_steps 8

# Reduce max length
--max_length 128

# Use smaller model
--model microsoft/mpnet-base

# Enable gradient checkpointing
model.gradient_checkpointing_enable()
```

### Slow Training

**Symptoms**:
- <10 iterations/second
- High CPU usage

**Solutions**:
```bash
# Use GPU
--device cuda

# Increase batch size
--batch_size 16

# Reduce logging frequency
--log_interval 50

# Use mixed precision
--fp16

# Disable gradient checkpointing
```

### Poor Convergence

**Symptoms**:
- Loss not decreasing
- F1 stuck at low value

**Solutions**:
```bash
# Increase learning rate
--lr 1e-5

# Reduce weight decay
--weight_decay 1e-3

# Increase warmup
--warmup_ratio 0.2

# Check data balance
# Ensure 60-40 or 65-35 ratio

# Try different optimizer
optimizer = torch.optim.Adam(...)
```

### Overfitting

**Symptoms**:
- Train F1 >> Dev F1
- Dev loss increasing

**Solutions**:
```bash
# Increase dropout
DROPOUT_1 = 0.4
DROPOUT_2 = 0.3

# Increase weight decay
--weight_decay 1e-1

# Reduce model size
--model microsoft/mpnet-base

# Add more data
USE_SYNTHETIC = True

# Early stopping
--patience 3
```

### Underfitting

**Symptoms**:
- Train F1 << Expected
- Both train and dev F1 low

**Solutions**:
```bash
# Increase model capacity
--model microsoft/deberta-v3-large

# Increase epochs
--epochs 30

# Reduce regularization
--weight_decay 1e-4
DROPOUT_1 = 0.2

# Check data quality
# Verify labels are correct
```

### Distribution Mismatch

**Symptoms**:
- High precision, low recall (or vice versa)
- Many false positives/negatives

**Solutions**:
```bash
# Use domain-only data (Track A)
--track A

# Balance classes
USE_CLASS_WEIGHTS = True

# Check test set distribution
# Ensure similar to training

# Add domain-specific data
# Collect more in-domain samples
```

---

## Best Practices

### 1. Start Simple
```bash
# Begin with frozen encoder
python scripts/classifier/finetune/train_mpnet_phase1_5_context_head_v3.py \
  --features full \
  --n-before 1 \
  --n-after 1

# Then try full fine-tuning
python scripts/classifier/finetune/finetune_deberta.py \
  --track A \
  --context_size 2
```

### 2. Use Appropriate Data
- **Track A** for domain-specific deployment
- **Track B** for general-purpose model
- **Domain-only** outperforms mixed-source

### 3. Monitor Metrics
- Primary: F1-score
- Secondary: Accuracy, ROC-AUC
- Watch: Precision/Recall balance

### 4. Save Everything
```python
# Save config
json.dump(config, open("config.json", "w"))

# Save checkpoints
torch.save(checkpoint, "best.pt")
torch.save(checkpoint, "last.pt")

# Save history
json.dump(history, open("history.json", "w"))
```

### 5. Reproducibility
```python
# Fix all seeds
set_seed(42)

# Document environment
pip freeze > requirements.txt

# Save git commit
git rev-parse HEAD > commit.txt
```

---

## References

- **Evaluation Guide**: [EVALUATION.md](EVALUATION.md)
- **Scripts Reference**: [SCRIPTS.md](SCRIPTS.md)
- **Dataset Guide**: [DATASET.md](DATASET.md)
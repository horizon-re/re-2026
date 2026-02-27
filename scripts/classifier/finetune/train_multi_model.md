# Multi-Task DeBERTa Training Guide

## Overview

This script trains multi-task models for requirement classification using:
- **Task 1 (Primary)**: Requirement vs Non-Requirement
- **Task 2 (Secondary)**: Functional vs Non-Functional  
- **Task 3 (Tertiary)**: Ambiguity Detection

## Data Sources

1. **Contextual Requirements**: `classifier/outputs/splits/holdout_60.jsonl`
   - Contains requirements with context windows
   - Labels: `requirement`, `non_requirement`, `with_context`, `ambiguous`, `non_ambiguous`

2. **F/NFR Data**: `classifier/outputs/splits_frnfr/train.jsonl` & `dev.jsonl`
   - Functional vs Non-Functional requirement labels
   - All samples are requirements (used for Task 2)

3. **Synthetic Non-Requirements**: `classifier/synthetic/non_requirements/synthetic_non_requirements.jsonl`
   - 5,000+ synthetic non-requirement examples
   - Helps balance the dataset

## Quick Start

### 1. Basic Training (DeBERTa, contextual data only)
```bash
python train_multitask_deberta.py \
  --model deberta \
  --n-context 1
```

### 2. Full Multi-Task Training (All Data Sources)
```bash
python train_multitask_deberta.py \
  --model deberta \
  --n-context 1 \
  --use-frnfr \
  --use-synthetic \
  --epochs 12 \
  --patience 4
```

### 3. Try Different Models

**DeBERTa (Recommended - Best for Classification)**
```bash
python train_multitask_deberta.py --model deberta --use-frnfr --use-synthetic
```

**MiniLM (Fast, Good Performance)**
```bash
python train_multitask_deberta.py --model minilm --use-frnfr --use-synthetic
```

**MPNet Original**
```bash
python train_multitask_deberta.py --model mpnet --use-frnfr --use-synthetic
```

**Paraphrase MPNet**
```bash
python train_multitask_deberta.py --model paraphrase-mpnet --use-frnfr --use-synthetic
```

## Advanced Training Options

### Hyperparameter Tuning

**Higher LoRA Rank (More Capacity)**
```bash
python train_multitask_deberta.py \
  --model deberta \
  --lora-r 32 \
  --lora-alpha 64 \
  --use-frnfr \
  --use-synthetic
```

**Adjusted Learning Rate**
```bash
python train_multitask_deberta.py \
  --model deberta \
  --lr 3e-5 \
  --use-frnfr \
  --use-synthetic
```

**Larger Batch Size**
```bash
python train_multitask_deberta.py \
  --model deberta \
  --batch-size 32 \
  --use-frnfr \
  --use-synthetic
```

**Longer Training**
```bash
python train_multitask_deberta.py \
  --model deberta \
  --epochs 20 \
  --patience 5 \
  --use-frnfr \
  --use-synthetic
```

### Task Weighting

Adjust importance of different tasks:

```bash
python train_multitask_deberta.py \
  --model deberta \
  --task1-weight 1.0 \   # Requirement classification (primary)
  --task2-weight 0.5 \   # F/NFR classification (secondary)
  --task3-weight 0.3 \   # Ambiguity detection (tertiary)
  --use-frnfr \
  --use-synthetic
```

## Recommended Experiments

### Experiment 1: Baseline DeBERTa
```bash
python train_multitask_deberta.py \
  --model deberta \
  --n-context 1 \
  --use-frnfr \
  --use-synthetic \
  --seed 42
```

### Experiment 2: Higher Capacity
```bash
python train_multitask_deberta.py \
  --model deberta \
  --n-context 1 \
  --use-frnfr \
  --use-synthetic \
  --lora-r 32 \
  --lora-alpha 64 \
  --seed 42
```

### Experiment 3: Longer Context
```bash
python train_multitask_deberta.py \
  --model deberta \
  --n-context 2 \
  --use-frnfr \
  --use-synthetic \
  --seed 42
```

### Experiment 4: Model Comparison (Run all 4)
```bash
# DeBERTa
python train_multitask_deberta.py --model deberta --use-frnfr --use-synthetic --seed 42

# MiniLM
python train_multitask_deberta.py --model minilm --use-frnfr --use-synthetic --seed 42

# MPNet
python train_multitask_deberta.py --model mpnet --use-frnfr --use-synthetic --seed 42

# Paraphrase-MPNet
python train_multitask_deberta.py --model paraphrase-mpnet --use-frnfr --use-synthetic --seed 42
```

## Understanding Output

### Training Metrics
```
[ep 01] req_f1=0.740 req_acc=0.590 | func_f1=0.680 | amb_f1=0.620
        ^^^^^^^^      ^^^^^^^^        ^^^^^^^^        ^^^^^^^^
        Primary       Primary         Secondary       Tertiary
        metric        metric          metric          metric
```

### Output Structure
```
classifier/models/multitask_deberta_lora/
└── 20260207_123456/                    # Run timestamp
    ├── run_meta.json                   # Run metadata
    └── deberta/                        # Model type folder
        ├── config.json                 # Model config
        ├── best.pt                     # Best checkpoint (by req_f1)
        ├── last.pt                     # Last checkpoint
        └── history.json                # Training history
```

## Expected Performance

Based on your current baseline (79.6% F1 with MPNet):

| Model | Expected F1 | Speed | Notes |
|-------|-------------|-------|-------|
| **DeBERTa** | **82-85%** | Slow | Best accuracy, worth the wait |
| MiniLM | 80-82% | Fast | Good balance |
| MPNet | 80-83% | Medium | Solid choice |
| Paraphrase-MPNet | 80-82% | Medium | Good for semantic similarity |

## Tips for Best Results

1. **Always use `--use-frnfr` and `--use-synthetic`**: More data = better performance
2. **Start with DeBERTa**: It's typically the best for classification tasks
3. **Monitor Task 1 (req_f1)**: This is your primary metric
4. **Task 2 (func_f1) helps Task 1**: Multi-task learning improves primary task
5. **Check class balance**: Review the output for class distribution
6. **Use multiple seeds**: Run with seeds 42, 123, 456 and average results

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python train_multitask_deberta.py --batch-size 8 --model deberta --use-frnfr --use-synthetic
```

### Slow Training
```bash
# Use MiniLM instead
python train_multitask_deberta.py --model minilm --use-frnfr --use-synthetic
```

### Poor Performance
1. Check if data loaded correctly (check logs)
2. Try higher LoRA rank: `--lora-r 32 --lora-alpha 64`
3. Try longer training: `--epochs 15 --patience 5`
4. Check for class imbalance in logs

## Next Steps After Training

1. **Evaluate on test set**: Use the best model on your held-out test data
2. **Ensemble models**: Combine predictions from multiple runs
3. **Threshold tuning**: Optimize classification threshold on dev set
4. **Error analysis**: Review misclassified examples

## Full Arguments Reference

```
--model {deberta,minilm,mpnet,paraphrase-mpnet}
                      Base model to use (default: deberta)
--n-context INT       Context window size (default: 1)
--use-frnfr          Include F/NFR training data
--use-synthetic      Include synthetic non-requirements
--seed INT           Random seed (default: 42)
--device {cuda,cpu}  Device to use
--epochs INT         Number of epochs (default: 10)
--patience INT       Early stopping patience (default: 3)
--batch-size INT     Batch size (default: 16)
--dev-ratio FLOAT    Dev set ratio (default: 0.15)
--lr FLOAT           Learning rate (default: 2e-5)
--weight-decay FLOAT Weight decay (default: 1e-4)
--grad-clip FLOAT    Gradient clipping (default: 1.0)
--lora-r INT         LoRA rank (default: 16)
--lora-alpha INT     LoRA alpha (default: 32)
--lora-dropout FLOAT LoRA dropout (default: 0.1)
--task1-weight FLOAT Weight for requirement task (default: 1.0)
--task2-weight FLOAT Weight for F/NFR task (default: 0.5)
--task3-weight FLOAT Weight for ambiguity task (default: 0.3)
```
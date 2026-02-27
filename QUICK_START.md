# Quick Start Guide

Get up and running with the Context-Aware Requirements Identification artifact in minutes.

## ⚡ 5-Minute Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/horizon-re/re-2026.git
cd re-2026

# Install dependencies
pip install torch transformers sentence-transformers
pip install spacy stanza scikit-learn scipy numpy pandas tqdm pyyaml

# Download NLP models
python -m spacy download en_core_web_sm
```

### 2. Explore the Data
```bash
# View raw documents
ls 02_raw_requirements/fintech/
ls 02_raw_requirements/saas/

# Check processed data
ls classifier/outputs/splits/
```

### 3. Run a Quick Test
```python
# Load and inspect training data
import json

with open('classifier/outputs/splits/train_4k.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

print(f"Total samples: {len(data)}")
print(f"First sample: {data[0]}")
```

### 4. Train a Model (Optional)
```bash
# Train DeBERTa on Track A (takes ~2 hours on GPU)
python scripts/classifier/finetune/finetune_deberta.py \
  --track A \
  --context_size 2 \
  --epochs 15
```

## 📚 What to Read First

### If you want to...

**Understand the research**
→ Read [README.md](README.md) and [main.tex](main.tex)

**Process your own data**
→ Read [docs/PIPELINE.md](docs/PIPELINE.md)

**Train models**
→ Read [docs/TRAINING.md](docs/TRAINING.md)

**Evaluate models**
→ Read [docs/EVALUATION.md](docs/EVALUATION.md)

**Understand the dataset**
→ Read [docs/DATASET.md](docs/DATASET.md)

**Use specific scripts**
→ Read [docs/SCRIPTS.md](docs/SCRIPTS.md)

## 🎯 Common Tasks

### Task 1: Process Raw Documents
```bash
# Stage 0: Ingest documents
python scripts/requirement-unitizer/scripts/step_0_ingest.py

# Stage 1: Extract text
python scripts/classifier/dataset_crawler.py

# Stage 2: Segment sentences
python scripts/classifier/extract_sentences_spacy_stanza.py

# Stage 3: Filter and embed
python scripts/classifier/embedding_aggregator.py
```

### Task 2: Annotate Data
```bash
# LLM-assisted annotation (requires API key)
export DEEPSEEK_API_KEY="your-key"
python scripts/requirement-unitizer/llm_labeling/step_2_llm_reviewer.py

# Manual annotation (interactive)
python scripts/requirement-unitizer/annotation_ui/annotate_requirements.py
```

### Task 3: Train Models
```bash
# Frozen encoder baseline
python scripts/classifier/finetune/train_mpnet_phase1_5_context_head_v3.py \
  --context_size 1

# Full fine-tuning (Track A)
python scripts/classifier/finetune/finetune_deberta.py \
  --track A \
  --context_size 2

# Full fine-tuning (Track B)
python scripts/classifier/finetune/finetune_deberta.py \
  --track B \
  --context_size 2
```

### Task 4: Evaluate Results
```bash
# Evaluate DeBERTa
python scripts/classifier/eval/eval_deberta_comp.py \
  --model_path classifier/models/deberta_trackA_k2/best_model.pt

# Evaluate MPNet baseline
python scripts/classifier/eval/eval_mpnet_phase1_5_holdout.py \
  --model_path classifier/models/mpnet_phase1_5_k1/classifier_head.pt
```

## 🔍 Repository Navigation

```
re-2026/
├── README.md                    ← Start here
├── docs/                        ← Complete documentation
│   ├── README.md               ← Documentation index
│   ├── PIPELINE.md             ← Data processing
│   ├── SCRIPTS.md              ← Script reference
│   ├── DATASET.md              ← Dataset guide
│   ├── TRAINING.md             ← Model training
│   └── EVALUATION.md           ← Evaluation guide
├── 02_raw_requirements/        ← Raw documents
├── classifier/                 ← Pipeline outputs
│   ├── outputs/splits/         ← Training data
│   └── models/                 ← Trained models
└── scripts/                    ← All scripts
    ├── classifier/             ← Core pipeline
    └── requirement-unitizer/   ← Annotation tools
```

## 💡 Key Concepts

### Context Window
```
k=2 example:
[s_{i-2}] [s_{i-1}] [CTX] [target sentence] [CTX] [s_{i+1}] [s_{i+2}]
```

### Label Taxonomy
- `requirement`: Clear standalone requirement
- `non_requirement`: Not a requirement
- `with_context`: Requires context for identification
- `ambiguous`: Unclear even with context

### Experimental Tracks
- **Track A**: 4K domain-only samples (F1=0.894)
- **Track B**: 15K mixed-source samples (F1=0.883)

## 📊 Expected Results

### Frozen Encoder (MPNet)
```
Context k=0: F1=0.664, Acc=0.621
Context k=1: F1=0.824, Acc=0.794  (+16.0 F1)
Context k=2: F1=0.820, Acc=0.791
```

### Full Fine-Tuning (DeBERTa, Track A)
```
Context k=1: F1=0.882, Acc=0.861
Context k=2: F1=0.894, Acc=0.875  (Best)
Context k=3: F1=0.885, Acc=0.865
```

## 🐛 Troubleshooting

### Import Errors
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Model Download Issues
```bash
export TRANSFORMERS_CACHE=.cache/transformers
export SENTENCE_TRANSFORMERS_HOME=.cache/sentence_transformers
```

### GPU Memory Issues
```bash
# Reduce batch size
python script.py --batch_size 4 --grad_accum 8
```

### API Rate Limits
```python
# Add delays in scripts
import time
time.sleep(0.5)
```

## 🎓 Learning Path

### Beginner
1. Read [README.md](README.md)
2. Explore dataset in `classifier/outputs/splits/`
3. Run evaluation scripts
4. Review results

### Intermediate
1. Read [docs/PIPELINE.md](docs/PIPELINE.md)
2. Process sample documents
3. Train frozen encoder baseline
4. Compare with paper results

### Advanced
1. Read [docs/SCRIPTS.md](docs/SCRIPTS.md)
2. Run complete pipeline
3. Train full fine-tuning models
4. Extend with custom features

## 🚀 Next Steps

After quick start:
1. Read full documentation in [docs/](docs/)
2. Try different configurations
3. Experiment with your own data

## 📝 Cheat Sheet

### Essential Commands
```bash
# Process data
python scripts/classifier/dataset_crawler.py

# Segment sentences
python scripts/classifier/extract_sentences_spacy_stanza.py

# Train model
python scripts/classifier/finetune/finetune_deberta.py --track A --context_size 2

# Evaluate
python scripts/classifier/eval/eval_deberta_comp.py
```

### Essential Files
```
classifier/outputs/splits/train_4k.jsonl     # Training data
classifier/outputs/splits/test_rest.jsonl    # Test data
classifier/models/                           # Model checkpoints
docs/                                        # Documentation
```

### Essential Scripts
```
scripts/classifier/dataset_crawler.py                    # Text extraction
scripts/classifier/extract_sentences_spacy_stanza.py     # Segmentation
scripts/classifier/embedding_aggregator.py               # Filtering
scripts/classifier/finetune/finetune_deberta.py         # Training
scripts/classifier/eval/eval_deberta_comp.py            # Evaluation
```


## 📚 Additional Resources

- **Full Documentation**: [docs/](docs/)
- **Dataset Guide**: [docs/DATASET.md](docs/DATASET.md)
- **Script Reference**: [docs/SCRIPTS.md](docs/SCRIPTS.md)
- **Pipeline Guide**: [docs/PIPELINE.md](docs/PIPELINE.md)
- **Training Guide**: [docs/TRAINING.md](docs/TRAINING.md)
- **Evaluation Guide**: [docs/EVALUATION.md](docs/EVALUATION.md)

---

**Ready to dive deeper?** Check out the [complete documentation](docs/README.md)!

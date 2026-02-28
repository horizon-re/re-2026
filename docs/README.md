# Documentation Index

This guide documents the training methodology used in the study. Full replication may require GPUs, local models, external APIs, and environment configurations beyond the scope of anonymous review. The documentation is provided for transparency rather than out-of-the-box execution.

Welcome to the complete documentation for the Context-Aware Requirements Identification research artifact.

## 📚 Documentation Structure

### Core Documentation

1. **[PIPELINE.md](PIPELINE.md)** - Complete Data Processing Pipeline
   - Stage-by-stage workflow
   - Input/output specifications
   - Usage examples
   - Quality assurance

2. **[SCRIPTS.md](SCRIPTS.md)** - Scripts Reference
   - All scripts documented
   - Usage examples
   - Configuration options
   - Troubleshooting

3. **[DATASET.md](DATASET.md)** - Dataset Guide
   - Dataset structure
   - Format specifications
   - Label taxonomy
   - Usage guidelines

4. **[TRAINING.md](TRAINING.md)** - Model Training Guide
   - Frozen encoder, LoRA, and full fine-tuning
   - Hyperparameter configurations
   - Training monitoring and troubleshooting

5. **[EVALUATION.md](EVALUATION.md)** - Evaluation Guide
   - Metrics definitions and protocols
   - Ablation study methodology
   - Reproducing paper results
   - Error analysis

## 🚀 Quick Start

### For First-Time Users

1. Start with the main [README.md](../README.md) for project overview
2. Read [PIPELINE.md](PIPELINE.md) to understand the workflow
3. Check [DATASET.md](DATASET.md) for data format details

### For Researchers

1. Understand the methodology: [PIPELINE.md](PIPELINE.md)
2. Explore the dataset: [DATASET.md](DATASET.md)
3. Train models: [TRAINING.md](TRAINING.md)
4. Evaluate results: [EVALUATION.md](EVALUATION.md)

### For Developers

1. Check [SCRIPTS.md](SCRIPTS.md) for script documentation
2. See [PIPELINE.md](PIPELINE.md) for workflow details
3. Review [TRAINING.md](TRAINING.md) for model training
4. Check [EVALUATION.md](EVALUATION.md) for evaluation protocols

## 📖 Documentation by Topic

### Data Processing
- [PIPELINE.md](PIPELINE.md) - Complete pipeline workflow
- [SCRIPTS.md](SCRIPTS.md) - Processing scripts
- [DATASET.md](DATASET.md) - Data formats

### Model Training
- [TRAINING.md](TRAINING.md) - Training configurations and hyperparameters
- [SCRIPTS.md](SCRIPTS.md) - Training scripts

### Evaluation
- [EVALUATION.md](EVALUATION.md) - Metrics, ablation studies, error analysis
- [SCRIPTS.md](SCRIPTS.md) - Evaluation scripts

### Service Integration
- [SCRIPTS.md](SCRIPTS.md) - Service scripts

## 🔍 Finding Information

### By Task

**I want to...**

- **Process raw documents** → [PIPELINE.md](PIPELINE.md) Stage 0-3
- **Annotate data** → [PIPELINE.md](PIPELINE.md) Stage 4
- **Understand data format** → [DATASET.md](DATASET.md)
- **Train models** → [TRAINING.md](TRAINING.md)
- **Evaluate models** → [EVALUATION.md](EVALUATION.md)
- **Use a specific script** → [SCRIPTS.md](SCRIPTS.md)

### By Component

**I need information about...**

- **Sentence segmentation** → [PIPELINE.md](PIPELINE.md) Stage 1
- **Candidate detection** → [PIPELINE.md](PIPELINE.md) Stage 2
- **Semantic filtering** → [PIPELINE.md](PIPELINE.md) Stage 3
- **LLM annotation** → [PIPELINE.md](PIPELINE.md) Stage 4
- **Dataset splits** → [DATASET.md](DATASET.md) Experimental Tracks

### By File Type

**I'm working with...**

- **Raw documents** → [DATASET.md](DATASET.md) Raw Document Format
- **JSONL files** → [DATASET.md](DATASET.md) Data Format Specifications
- **Training splits** → [DATASET.md](DATASET.md) Training Split Format

## 📊 Research Artifacts

### Code
- **Scripts**: [SCRIPTS.md](SCRIPTS.md)
- **Pipeline**: [PIPELINE.md](PIPELINE.md)

### Data
- **Primary Corpus**: [DATASET.md](DATASET.md) Primary Corpus
- **External Datasets**: [DATASET.md](DATASET.md) External Datasets
- **Synthetic Data**: [DATASET.md](DATASET.md) Synthetic Data

## 🛠️ Common Tasks

### Setup and Installation
```bash
# See main README.md for installation
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Running the Pipeline
```bash
# See PIPELINE.md for complete workflow
./run_pipeline.sh
```

### Training Models
```bash
# See TRAINING.md for detailed instructions
python scripts/classifier/finetune/finetune_deberta.py --track A --context_size 2
```

### Evaluating Results
```bash
# See EVALUATION.md for evaluation guide
python scripts/classifier/eval/eval_deberta_comp.py
```

## 📝 Documentation Conventions

### Code Examples
All code examples are tested and runnable. Copy-paste should work directly.

### File Paths
All paths are relative to repository root unless otherwise specified.

### Commands
Commands assume you're in the repository root directory.

### Configuration
Default configurations are shown. Adjust as needed for your environment.

## 🔗 External Resources

### Dependencies
- **spaCy**: https://spacy.io/
- **Stanza**: https://stanfordnlp.github.io/stanza/
- **Transformers**: https://huggingface.co/docs/transformers/
- **Sentence-Transformers**: https://www.sbert.net/


## 📧 Support

### Getting Help

1. **Check Documentation**: Search this documentation first
2. **Review Examples**: Look at usage examples in each guide
3. **Check Issues**: Search GitHub issues for similar problems
4. **Ask Questions**: Open a new GitHub issue

### Reporting Issues

When reporting issues, include:
- Documentation page reference
- What you tried
- Expected vs actual behavior
- Error messages (if any)
- Environment details

## 📜 License

Documentation is released under MIT License. See [LICENSE](../LICENSE).
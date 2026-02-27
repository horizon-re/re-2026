# Documentation Index

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
   - Training configurations
   - Hyperparameters
   - Best practices
   - Reproducibility

5. **[EVALUATION.md](EVALUATION.md)** - Evaluation Guide
   - Metrics and protocols
   - Result interpretation
   - Comparison guidelines
   - Error analysis

6. **[API.md](API.md)** - API Reference
   - Service integrations
   - LLM clients
   - Utility functions
   - Configuration

## 🚀 Quick Start

### For First-Time Users

1. Start with the main [README.md](../README.md) for project overview
2. Read [PIPELINE.md](PIPELINE.md) to understand the workflow
3. Check [DATASET.md](DATASET.md) for data format details
4. Follow [TRAINING.md](TRAINING.md) to train models
5. Use [EVALUATION.md](EVALUATION.md) to evaluate results

### For Researchers

1. Review the research paper: [main.tex](../main.tex)
2. Understand the methodology: [PIPELINE.md](PIPELINE.md)
3. Explore the dataset: [DATASET.md](DATASET.md)
4. Reproduce experiments: [TRAINING.md](TRAINING.md)
5. Analyze results: [EVALUATION.md](EVALUATION.md)

### For Developers

1. Check [SCRIPTS.md](SCRIPTS.md) for script documentation
2. Review [API.md](API.md) for service integrations
3. See [PIPELINE.md](PIPELINE.md) for workflow details
4. Use [TRAINING.md](TRAINING.md) for model training
5. Refer to [EVALUATION.md](EVALUATION.md) for metrics

## 📖 Documentation by Topic

### Data Processing
- [PIPELINE.md](PIPELINE.md) - Complete pipeline workflow
- [SCRIPTS.md](SCRIPTS.md) - Processing scripts
- [DATASET.md](DATASET.md) - Data formats

### Model Training
- [TRAINING.md](TRAINING.md) - Training guide
- [SCRIPTS.md](SCRIPTS.md) - Training scripts
- [EVALUATION.md](EVALUATION.md) - Evaluation metrics

### Service Integration
- [API.md](API.md) - API reference
- [SCRIPTS.md](SCRIPTS.md) - Service scripts

## 🔍 Finding Information

### By Task

**I want to...**

- **Process raw documents** → [PIPELINE.md](PIPELINE.md) Stage 0-3
- **Annotate data** → [PIPELINE.md](PIPELINE.md) Stage 4
- **Train a model** → [TRAINING.md](TRAINING.md)
- **Evaluate results** → [EVALUATION.md](EVALUATION.md)
- **Understand data format** → [DATASET.md](DATASET.md)
- **Use a specific script** → [SCRIPTS.md](SCRIPTS.md)
- **Integrate LLM services** → [API.md](API.md)

### By Component

**I need information about...**

- **Sentence segmentation** → [PIPELINE.md](PIPELINE.md) Stage 1
- **Candidate detection** → [PIPELINE.md](PIPELINE.md) Stage 2
- **Semantic filtering** → [PIPELINE.md](PIPELINE.md) Stage 3
- **LLM annotation** → [PIPELINE.md](PIPELINE.md) Stage 4
- **Dataset splits** → [DATASET.md](DATASET.md) Experimental Tracks
- **Context windows** → [TRAINING.md](TRAINING.md) Context Construction
- **Evaluation metrics** → [EVALUATION.md](EVALUATION.md) Metrics

### By File Type

**I'm working with...**

- **Raw documents** → [DATASET.md](DATASET.md) Raw Document Format
- **JSONL files** → [DATASET.md](DATASET.md) Data Format Specifications
- **Training splits** → [DATASET.md](DATASET.md) Training Split Format
- **Model checkpoints** → [TRAINING.md](TRAINING.md) Model Outputs
- **Evaluation results** → [EVALUATION.md](EVALUATION.md) Results Format

## 📊 Research Artifacts

### Paper
- **LaTeX Source**: [main.tex](../main.tex)
- **Methodology**: Section 3 of paper
- **Evaluation**: Section 4 of paper
- **Results**: Section 4 of paper

### Code
- **Scripts**: [SCRIPTS.md](SCRIPTS.md)
- **Pipeline**: [PIPELINE.md](PIPELINE.md)
- **Training**: [TRAINING.md](TRAINING.md)

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

### Research Paper
- **Main Paper**: [main.tex](../main.tex)
- **Related Work**: Section 2 of paper
- **Methodology**: Section 3 of paper
- **Evaluation**: Section 4 of paper

### Dependencies
- **spaCy**: https://spacy.io/
- **Stanza**: https://stanfordnlp.github.io/stanza/
- **Transformers**: https://huggingface.co/docs/transformers/
- **Sentence-Transformers**: https://www.sbert.net/

### Datasets
- **DOSSPRE**: https://github.com/dosspre/dataset
- **HuggingFace**: https://huggingface.co/datasets

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

## 🤝 Contributing

### Documentation Improvements

We welcome documentation improvements:
- Clarifications
- Additional examples
- Error corrections
- New sections

Submit via GitHub pull request.

## 📜 License

Documentation is released under MIT License. See [LICENSE](../LICENSE).

## 🙏 Acknowledgments

This documentation was created to support reproducible research in requirements engineering.

---

**Last Updated**: February 2026  
**Version**: 1.0  
**Status**: Research Artifact for RE 2026

# Investigating the Representation of Backchannels and Fillers in Fine-tuned Language Models

## Overview

The project is structured to explore the representation spaces of backchannels/fillers (discourse markers) using distinct language modeling paradigms:
- **NTP_and_MASK Models**: Evaluating generic causal and masked language models (BERT, GPT-2, LLaMA-3, Qwen-3) using Next Token Prediction and Masked Language Modeling.
- **TurnGPT**: Evaluating fine-tuned models specifically designed for turn-taking projection (GPT-2 based LM).

The analytical pipeline includes extracting model embeddings across English and Japanese, clustering them via K-Means, evaluating clustering quality with Silhouette Scores, and exploring high-dimensional representations with t-SNE and distance matrices. Further analysis includes NLG evaluation using metrics such as BERTScore, Perplexity etc.

## Dialogue Datasets (open-sourced dataset)
- **English Dialogue**: Switchboard, Map Task
- **Japanese Dialogue**: BTSJ
  
## Directory Structure

```text
.
├── NTP_and_MASK/           # Experiments with standard Next Token Prediction and Masked LMs
│   ├── data/               # English and Japanese dialogue datasets
│   ├── utils/              # Shared utilities (embeddings extraction, K-Means, metrics)
├── TurnGPT/                # Experiments using Turn-taking GPT models
│   ├── dataset/            # TurnGPT-specific formatted datasets
│   ├── datasets_turntaking/# Dialog data processing packages
│   └── turngpt_discourse_marker/
│                           # TurnGPT specific backchannel/filler evaluation scripts
└── bootstrap_results.ipynb # Statistical bootstrapping and significance testing
```

## Supported Models & Languages
- **Models**: BERT, GPT-2, LLaMA-3, Qwen-3
- **Languages**: English (e.g., Switchboard and MapTask), Japanese (e.g., BTSJ)

## Experimental Pipeline

1. **Dataset Preparation**:
   Split English/Japanese dialogue datasets using `split_dataset.ipynb`.
2. **Feature Extraction**:
   Extract hidden states (embeddings) for target fillers, backchannels, and context via `utils/get_embedding.py`.
3. **Clustering Analysis**:
   Apply K-Means clustering algorithm on extracted representations using `utils/k_means.py`.
4. **Inference for NLG**:
   Perform inference to generate utterances and evaluate NLG metrics using `infer.py`.


## Installation

**NTP/MASK Experiments**:
Basic requirements include PyTorch, Transformers, scikit-learn, and standard data science libraries (numpy, pandas, matplotlib, seaborn).

**TurnGPT Experiments**:
Note that `TurnGPT` has its own setup requirements. Please follow `TurnGPT/README.md` for installation.

## Usage

**Running NTP_and_MASK Experiments**
```bash
cd NTP_and_MASK
# Finetuning / Model Preparation
python train.py --language Japanese --pretrainModel gpt2          
# Inference for Natural Language Generation (NLG) task
python infer.py --language English --pretrainModel llama3 --test_type no_ft_one --infer_ratio 10      
# Metrics Evaluation and K-Means clustering
python test.py --language Japanese --pretrainModel bert --test_type no_ft_no --pca             
```

**Running TurnGPT Experiments**
```bash
cd TurnGPT
# TurnGPT model training
bash script_train.bash              
cd turngpt_discourse_marker
# Clustering and Representation analysis
python turngpt_discourse_marker/test.py --language Japanese --test_type ft_one --pca    
```

<!-- ---

## Citation

```bibtex
@inproceedings{tpp2026timing,
  title     = {Timing of Information Can Distinguish Human and {LLM} Generations: Evidence from Temporal Point Processes},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026},
}
``` -->

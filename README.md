# SigExt Extensions

This repository extends the original SigExt framework:
https://github.com/amazon-science/SigExt

It adds:
- Graph-based evidence selection (GraphSigExt)
- Multilingual summarization
- Semantic-label keyphrase extraction
- Support for Claude (AWS Bedrock) and Llama (HuggingFace)

Experiments are conducted on CNN/DailyMail.

## Repository Structure

```text
.
├── src/
│   ├── prepare_data.py
│   ├── train_longformer_extractor_context.py
│   ├── inference_longformer_extractor.py
│   ├── train_longformer_extractor_context_semantic.py
│   ├── inference_longformer_extractor_semantic.py
│   ├── zs_summarization.py
│   ├── prompts.py
│   └── bedrock_utils.py
│
└── experiments/
    └── (generated datasets, checkpoints, predictions, summaries, metrics)
```

All generated outputs are stored under `experiments/`.

## Installation

Recommended: Python 3.10+

### Dependencies

```bash
pip install torch pytorch-lightning \
transformers datasets \
numpy tqdm jsonlines \
nltk rapidfuzz \
rouge-score bert-score \
sentence-transformers \
boto3
```

Download NLTK tokenizer:

```bash
python -c "import nltk; nltk.download('punkt')"
```

## Backend Configuration

### Claude (AWS Bedrock)

Set environment variables:

```bash
export SONNET4_MODEL_ID_OR_ARN="YOUR_MODEL_ID"
export SONNET4_REGION="eu-north-1"

export CLAUDE_MAX_TOKENS=512
export CLAUDE_TEMPERATURE=1.0
export CLAUDE_TOP_P=0.8
```

Run with:

```bash
--model_name claude
```

### Llama (HuggingFace)

Default model:

```text
meta-llama/Llama-3.2-1B-Instruct
```

Optional override:

```bash
export LLAMA_MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
```

Run with:

```bash
--model_name llama
```

## Data Preparation (CNN/DailyMail)

```bash
python src/prepare_data.py \
  --dataset cnn \
  --output_dir experiments/cnn_dataset
```

Creates:

```text
experiments/cnn_dataset/
  train.jsonl
  validation.jsonl
  test.jsonl
```

## Keyphrase Extraction

### Baseline Extractor

Train:

```bash
python src/train_longformer_extractor_context.py \
  --dataset_dir experiments/cnn_dataset \
  --checkpoint_dir experiments/cnn_extractor_model
```

Inference:

```bash
python src/inference_longformer_extractor.py \
  --dataset_dir experiments/cnn_dataset \
  --checkpoint_dir experiments/cnn_extractor_model \
  --output_dir experiments/cnn_keyphrases
```

### Semantic Extractor

Uses sentence-transformer similarity for label assignment.

Train:

```bash
python src/train_longformer_extractor_context_semantic.py \
  --dataset_dir experiments/cnn_dataset \
  --checkpoint_dir experiments/cnn_extractor_model_semantic \
  --semantic_threshold 0.7 \
  --semantic_model_name all-MiniLM-L6-v2 \
  --semantic_device cuda
```

Inference:

```bash
python src/inference_longformer_extractor_semantic.py \
  --dataset_dir experiments/cnn_dataset \
  --checkpoint_dir experiments/cnn_extractor_model_semantic \
  --output_dir experiments/cnn_keyphrases_semantic
```

## Summarization

All summarization is performed with:
`src/zs_summarization.py`

Evaluation metrics:
- ROUGE-1 (R/F)
- ROUGE-L (F)
- BERTScore (P/R/F1)

### Method Selection

Controlled via `--kw_strategy`:

| Strategy       | Description                                 |
|----------------|---------------------------------------------|
| `disable`      | Naive baseline (no keyphrases)              |
| `sigext_topk`  | SigExt (top-k keyphrases)                   |
| `graph_sigext` | GraphSigExt (keyphrases + evidence graph)   |

### Baseline (Naive)

```bash
python src/zs_summarization.py \
  --dataset cnn \
  --dataset_dir experiments/cnn_dataset \
  --output_dir experiments/runs/naive_llama_en \
  --model_name llama \
  --kw_strategy disable \
  --target_lang en \
  --bertscore_model roberta-large
```

### SigExt

```bash
python src/zs_summarization.py \
  --dataset cnn \
  --dataset_dir experiments/cnn_dataset \
  --output_dir experiments/runs/sigext_llama_en \
  --model_name llama \
  --kw_strategy sigext_topk \
  --kw_model_top_k 15 \
  --target_lang en
```

### GraphSigExt

Additional graph parameters:
- `--graph_sent_budget`
- `--graph_window`
- `--graph_match_threshold`
- `--graph_terminals`

Example:

```bash
python src/zs_summarization.py \
  --dataset cnn \
  --dataset_dir experiments/cnn_dataset \
  --output_dir experiments/runs/graphsigext_llama_en \
  --model_name llama \
  --kw_strategy graph_sigext \
  --kw_model_top_k 15 \
  --graph_sent_budget 10 \
  --graph_window 2 \
  --graph_match_threshold 70 \
  --graph_terminals 10 \
  --target_lang en
```

## Multilingual Summaries

Multilingual mode is activated when:
`--target_lang != en`

The system automatically switches to multilingual prompt templates defined in `src/prompts.py`.

Example (Italian summaries):

```bash
python src/zs_summarization.py \
  --dataset cnn \
  --dataset_dir experiments/cnn_dataset \
  --output_dir experiments/runs/graphsigext_claude_it \
  --model_name claude \
  --kw_strategy graph_sigext \
  --kw_model_top_k 20 \
  --target_lang it \
  --bertscore_model xlm-roberta-large
```

## Experiment Flow

1. Prepare dataset
2. Train extractor (baseline or semantic)
3. Run extractor inference
4. Run summarization with desired:
   - model (Claude / Llama)
   - strategy (naive / sigext / graph)
   - language (English or multilingual)

All outputs are saved under `experiments/`.

## Upstream Reference

This repository extends:
SigExt
https://github.com/amazon-science/SigExt


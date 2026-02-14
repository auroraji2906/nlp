# NLP Experiments (SigExt Extensions)

This repository contains experiments built on top of SigExt for prompt-based abstractive summarization.

## Installation

### 1. Requirements

Recommended: Python 3.10+

```bash
pip install torch pytorch-lightning transformers datasets \
numpy tqdm jsonlines nltk rapidfuzz rouge-score \
matplotlib pandas seaborn rake-nltk boto3
```

Notes:
- `bert-score` and `sentence-transformers` are optional for external evaluation workflows.
- Some models may require extra system/CUDA dependencies depending on your environment.

### 2. NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Backend Configuration

### Claude (AWS Bedrock)

For the current implementation in `src/bedrock_utils.py`, model and region are hardcoded:
- Region: `us-west-2`
- Model ID: `anthropic.claude-instant-v1`

Make sure your AWS credentials are configured for Bedrock access in that region.

### Llama (HuggingFace)

The default local model in `src/bedrock_utils.py` is:
- `meta-llama/Llama-3.2-1B-Instruct`

If you want a different model name (for example `meta-llama/Llama-3.2-3B-Instruct`), update `LLAMA_MODEL_NAME` in `src/bedrock_utils.py`.

## Experiment Flow

### 1. Data Preparation

```bash
python src/prepare_data.py --dataset cnn --output_dir experiments/cnn_dataset
```

### 2. Keyphrase Extraction

Train extractor:

```bash
python src/train_longformer_extractor_context.py \
  --dataset_dir experiments/cnn_dataset \
  --checkpoint_dir experiments/cnn_extractor_model
```

Run extractor inference to produce `input_kw_model` scores:

```bash
python src/inference_longformer_extractor.py \
  --dataset_dir experiments/cnn_dataset \
  --checkpoint_dir experiments/cnn_extractor_model \
  --output_dir experiments/cnn_dataset_with_keyphrase
```

### 3. Run Summarization

Choose strategy with `--kw_strategy`:
- `disable`: Prompt + document only
- `sigext_topk`: Standard SigExt top-k keyphrases
- `graph_sigext`: Keyphrases + graph-based evidence selection

Example (`graph_sigext`, Claude):

```bash
python src/zs_summarization.py \
  --dataset cnn \
  --dataset_dir experiments/cnn_dataset_with_keyphrase \
  --output_dir experiments/cnn_extsig_graph_predictions \
  --model_name claude \
  --kw_strategy graph_sigext \
  --kw_model_top_k 20
```

## Metrics and Outputs

Results are saved under `experiments/`, including:
- Predictions JSON
- Processed dataset with prompts
- ROUGE metrics (`*_metrics.json`)

## References

- Upstream repository: https://github.com/amazon-science/sigext
- Paper: https://www.amazon.science/publications/salient-information-prompting-to-steer-content-in-prompt-based-abstractive-summarization

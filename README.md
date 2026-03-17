# Standalone Perplexity Embedding Pipeline

This folder is self-contained so it can be copied into a new git repo and uploaded to a cloud machine.

## Folder Layout

- `run_pplx_embedding_pipeline.py`
- `requirements.txt`
- `data/train.parquet`
- `data/valid.parquet`
- `data/test.parquet`
- `output/`

## What It Does

The script:

- loads the split parquet files from `data/`
- builds deduplicated embeddings for:
  - `query_text`
  - `user_profile_text`
  - `candidate_text`
- persists embeddings with resume support
- writes compressed `.npz` embedding files for later download
- computes cosine features for `256`, `512`, and `1024` MRL prefixes
- writes feature-enriched split parquet files
- writes `progress.json` and checkpoint state files so interrupted runs can resume

## Outputs

Inside `output/`:

- `progress.json`
- `manifest.json`
- `store/query_texts.parquet`
- `store/query_embeddings.int8.npy`
- `store/query_embeddings.int8.npz`
- `store/user_texts.parquet`
- `store/user_embeddings.int8.npy`
- `store/user_embeddings.int8.npz`
- `store/candidate_texts.parquet`
- `store/candidate_embeddings.int8.npy`
- `store/candidate_embeddings.int8.npz`
- `feature_splits/train_with_pplx_features.parquet`
- `feature_splits/valid_with_pplx_features.parquet`
- `feature_splits/test_with_pplx_features.parquet`

The `.npy` files are checkpoint-friendly working files.
The `.npz` files are the compressed files meant for download and reuse later.

## Resume And Progress

Resume is implemented in two places:

- per text role:
  - `store/query_embedding_state.json`
  - `store/user_embedding_state.json`
  - `store/candidate_embedding_state.json`
- per feature split:
  - `feature_splits/train_feature_state.json`
  - `feature_splits/valid_feature_state.json`
  - `feature_splits/test_feature_state.json`

If the process stops, rerun the same command in the same folder and it will continue from the next unfinished batch or chunk.

Live progress is always written to:

- `output/progress.json`

## Install

Create a venv and install the right torch wheel for your machine first.

Example for NVIDIA GPU with CUDA 13.0-compatible wheel:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you want CPU only, install plain `torch` from PyPI instead.

## Run

Auto device selection:

```powershell
.\.venv\Scripts\python.exe run_pplx_embedding_pipeline.py
```

Force GPU:

```powershell
.\.venv\Scripts\python.exe run_pplx_embedding_pipeline.py --device cuda
```

Force CPU:

```powershell
.\.venv\Scripts\python.exe run_pplx_embedding_pipeline.py --device cpu
```

Set an explicit embedding batch size:

```powershell
.\.venv\Scripts\python.exe run_pplx_embedding_pipeline.py --batch-size 128
```

Switch between model sizes:

```powershell
.\.venv\Scripts\python.exe run_pplx_embedding_pipeline.py --model-size 0.6b
.\.venv\Scripts\python.exe run_pplx_embedding_pipeline.py --model-size 4b
```

## Defaults

- backend: `hf-local`
- model: `perplexity-ai/pplx-embed-v1-0.6B`
- model size preset: `0.6b`
- embedding dimensions: `1024`
- data dir: `./data`
- output dir: `./output`
- cache dir: `./hf-cache`
- MRL dims: `256,512,1024`

## Notes

- For cold-start rows where `user_profile_text == coldstart`, user-based cosine features are set to zero.
- The combined query+user candidate cosine falls back to query-only behavior for cold-start rows.
- The first run downloads the Hugging Face model into `hf-cache/`.
"# dataset_recruitors" 

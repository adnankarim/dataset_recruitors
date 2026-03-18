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

## Train XGBoost On Extracted Features

The training entrypoint is:

- `train_xgboost_on_pplx_features.py`

It trains on:

- `output/feature_splits/train_with_pplx_features.parquet`
- `output/feature_splits/valid_with_pplx_features.parquet`
- `output/feature_splits/test_with_pplx_features.parquet`

Expected label columns:

- target: `label`
- optional sample weight: `label_weight`

Detected target classes in this dataset:

- `Go`
- `Interesting`
- `Not interesting`
- `Out of scope`
- `Why not`

By default the trainer:

- uses only the Perplexity cosine feature columns as model inputs
- trains a multiclass XGBoost classifier on the `label` column
- leaves class-imbalance handling off unless you enable it
- checkpoints the model during training
- can resume from the latest checkpoint
- writes evaluation metrics, predictions, and plots

Train from scratch:

```powershell
.\.venv\Scripts\python.exe train_xgboost_on_pplx_features.py
```

Train on GPU:

```powershell
.\.venv\Scripts\python.exe train_xgboost_on_pplx_features.py --device cuda
```

Train with class-imbalance handling enabled:

```powershell
.\.venv\Scripts\python.exe train_xgboost_on_pplx_features.py --class-imbalance-handling balanced-sample-weight
```

Train with class-imbalance handling explicitly off:

```powershell
.\.venv\Scripts\python.exe train_xgboost_on_pplx_features.py --class-imbalance-handling off
```

Resume from the latest checkpoint:

```powershell
.\.venv\Scripts\python.exe train_xgboost_on_pplx_features.py --resume
```

Example with custom run directory and boosting rounds:

```powershell
.\.venv\Scripts\python.exe train_xgboost_on_pplx_features.py --run-dir training_runs\xgboost_exp_01 --num-boost-round 2000 --checkpoint-interval 100
```

### Training Outputs

Inside the run directory, for example `training_runs/xgboost_pplx/`:

- `artifacts/metrics.json`
- `artifacts/metrics.csv`
- `artifacts/feature_columns.json`
- `artifacts/label_mapping.json`
- `artifacts/preprocessing.json`
- `artifacts/run_summary.json`
- `predictions/train_predictions.parquet`
- `predictions/valid_predictions.parquet`
- `predictions/test_predictions.parquet`
- `models/xgboost_model_final.json`
- `checkpoints/model_round_*.json`
- `checkpoints/training_state.json`
- `plots/learning_curves.png`
- `plots/label_distribution.png`
- `plots/roc_curves.png`
- `plots/pr_curves.png`
- `plots/calibration_curves.png`
- `plots/prediction_histograms.png`
- `plots/confusion_matrix_valid.png`
- `plots/confusion_matrix_test.png`
- `plots/feature_importance.csv`
- `plots/feature_importance_gain_topk.png`
- `plots/feature_importance_weight_topk.png`

### Default Modeling Choices

- objective: multiclass classification with `label`
- input features: only `pplx_qc_*`, `pplx_uc_*`, `pplx_qu_*`, and `pplx_quc_*`
- class imbalance handling: `off` by default, optional `balanced-sample-weight`
- evaluation split for model selection: `valid`
- predictions: class probabilities plus top predicted label
- early stopping: enabled
- checkpoints: saved every fixed number of boosting rounds

### Class Imbalance Handling

The trainer handles class imbalance at the label level for the 5-class target:

- `Go`: `44388`
- `Interesting`: `12058`
- `Not interesting`: `18758`
- `Out of scope`: `18198`
- `Why not`: `18568`

End-to-end flow:

1. Read `label` from the train split and discover all classes.
2. Build a stable label mapping written to `artifacts/label_mapping.json`.
3. If `--class-imbalance-handling off` is used:
   the trainer uses only the original `label_weight` values if that column exists.
4. If `--class-imbalance-handling balanced-sample-weight` is used:
   the trainer computes per-class weights from the train split as `total_rows / (num_classes * class_count)`.
5. For each training row, the final training weight is:
   `class_weight(label) * label_weight` when `label_weight` exists, otherwise just `class_weight(label)`.
6. Those combined sample weights are passed into XGBoost during training and evaluation.
7. The selected imbalance mode and computed class weights are written to:
   `artifacts/run_summary.json` and `artifacts/label_mapping.json`.

What this does:

- gives more influence to underrepresented classes such as `Interesting`
- gives less influence to the largest class such as `Go`
- works for multiclass training, unlike binary-only `scale_pos_weight`

What it does not do:

- it does not resample rows
- it does not change labels
- it does not rebalance the validation or test files on disk

Recommended usage:

```powershell
.\.venv\Scripts\python.exe train_xgboost_on_pplx_features.py --class-imbalance-handling balanced-sample-weight
```

Disable it explicitly:

```powershell
.\.venv\Scripts\python.exe train_xgboost_on_pplx_features.py --class-imbalance-handling off
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

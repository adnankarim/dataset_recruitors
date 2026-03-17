import argparse
import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SPLITS_DIR = str(BASE_DIR / "data")
DEFAULT_OUTPUT_DIR = str(BASE_DIR / "output")
DEFAULT_MODEL = "pplx-embed-v1-0.6B"
DEFAULT_API_URL = "https://api.perplexity.ai/v1/embeddings"
DEFAULT_HF_MODEL_ID = "perplexity-ai/pplx-embed-v1-0.6B"
DEFAULT_HF_CACHE_DIR = str(BASE_DIR / "hf-cache")
MODEL_SIZE_PRESETS = {
    "0.6b": {
        "model": "pplx-embed-v1-0.6B",
        "hf_model_id": "perplexity-ai/pplx-embed-v1-0.6B",
    },
    "4b": {
        "model": "pplx-embed-v1-4b",
        "hf_model_id": "perplexity-ai/pplx-embed-v1-4b",
    },
}
SPLIT_NAMES = ("train", "valid", "test")
TEXT_SPECS = (
    ("query", "query_text"),
    ("user", "user_profile_text"),
    ("candidate", "candidate_text"),
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def update_progress(progress_path: Path, payload: dict) -> None:
    current = load_json_if_exists(progress_path) or {}
    current.update(payload)
    current["updated_at"] = utc_now()
    write_json(progress_path, current)


def parse_mrl_dims(value: str, embedding_dim: int) -> list[int]:
    dims = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        dim = int(part)
        if dim <= 0 or dim > embedding_dim:
            raise ValueError(f"Invalid MRL dimension {dim}; expected 1..{embedding_dim}.")
        dims.append(dim)
    dims = sorted(set(dims))
    if embedding_dim not in dims:
        dims.append(embedding_dim)
    return dims


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return parsed


def resolve_model_config(model_size: str | None, model: str, hf_model_id: str) -> tuple[str, str]:
    if not model_size:
        return model, hf_model_id
    preset = MODEL_SIZE_PRESETS[model_size]
    return preset["model"], preset["hf_model_id"]


def load_split_frames(splits_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for split_name in SPLIT_NAMES:
        path = splits_dir / f"{split_name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        frames[split_name] = pd.read_parquet(path)
    return frames


def build_unique_text_table(frames: dict[str, pd.DataFrame], column: str, role: str) -> pd.DataFrame:
    series_list = [frame[column].fillna("").astype(str) for frame in frames.values()]
    all_values = pd.concat(series_list, ignore_index=True)
    counts = all_values.value_counts(dropna=False)
    table = pd.DataFrame(
        {
            f"{role}_text_id": np.arange(len(counts), dtype=np.int32),
            column: counts.index.astype(str),
            "source_count": counts.values.astype(np.int32),
        }
    )
    return table


def decode_base64_int8(encoded: str, dimensions: int) -> np.ndarray:
    raw = base64.b64decode(encoded)
    arr = np.frombuffer(raw, dtype=np.int8)
    if arr.size != dimensions:
        raise ValueError(f"Expected {dimensions} dimensions, got {arr.size}.")
    return arr.copy()


def deterministic_mock_embedding(text: str, dimensions: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")
    rng = np.random.default_rng(seed)
    return rng.integers(-127, 128, size=dimensions, dtype=np.int16).astype(np.int8)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def download_hf_model(model_id: str, cache_dir: Path) -> str:
    from huggingface_hub import snapshot_download

    ensure_dir(cache_dir)
    local_path = snapshot_download(repo_id=model_id, cache_dir=str(cache_dir))
    return local_path


def load_local_sentence_transformer(model_path: str, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_path, trust_remote_code=True, device=device)


def request_embeddings(
    texts: list[str],
    api_key: str,
    model: str,
    dimensions: int,
    api_url: str,
    timeout_seconds: int,
) -> list[np.ndarray]:
    payload = json.dumps(
        {
            "input": texts,
            "model": model,
            "dimensions": dimensions,
            "encoding_format": "base64_int8",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        body = json.loads(response.read().decode("utf-8"))
    data = sorted(body["data"], key=lambda row: row["index"])
    return [decode_base64_int8(item["embedding"], dimensions) for item in data]


def embed_text_table(
    table: pd.DataFrame,
    text_column: str,
    role: str,
    output_dir: Path,
    model: str,
    dimensions: int,
    batch_size: int,
    timeout_seconds: int,
    mock_embeddings: bool,
    api_key: str | None,
    api_url: str,
    backend: str,
    hf_encoder,
    progress_path: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    ensure_dir(output_dir)
    table_path = output_dir / f"{role}_texts.parquet"
    embeddings_path = output_dir / f"{role}_embeddings.int8.npy"
    compressed_path = output_dir / f"{role}_embeddings.int8.npz"
    state_path = output_dir / f"{role}_embedding_state.json"

    existing_state = load_json_if_exists(state_path)
    if table_path.exists() and embeddings_path.exists() and (not existing_state or existing_state.get("completed")):
        restored_table = pd.read_parquet(table_path)
        restored_embeddings = np.load(embeddings_path, mmap_mode="r")
        update_progress(
            progress_path,
            {
                "phase": "embedding",
                "role": role,
                "role_status": "completed",
                "role_completed": int(len(restored_table)),
                "role_total": int(len(restored_table)),
                "message": f"{role} embeddings already available; reusing persisted store.",
            },
        )
        return restored_table, restored_embeddings

    table.to_parquet(table_path, index=False)
    texts = table[text_column].tolist()
    total = len(texts)
    next_index = 0

    if embeddings_path.exists():
        embeddings = open_memmap(embeddings_path, mode="r+", dtype=np.int8, shape=(total, dimensions))
    else:
        embeddings = open_memmap(embeddings_path, mode="w+", dtype=np.int8, shape=(total, dimensions))

    if existing_state:
        next_index = int(existing_state.get("next_index", 0))
    elif embeddings_path.exists():
        next_index = 0

    started_at = time.time()
    update_progress(
        progress_path,
        {
            "phase": "embedding",
            "role": role,
            "role_status": "running",
            "role_completed": int(next_index),
            "role_total": int(total),
            "message": f"Embedding {role} texts from index {next_index}.",
        },
    )

    for start in range(next_index, total, batch_size):
        end = min(start + batch_size, total)
        batch = texts[start:end]
        if mock_embeddings:
            batch_vectors = [deterministic_mock_embedding(text, dimensions) for text in batch]
        elif backend == "hf-local":
            encoded = hf_encoder.encode(batch, batch_size=len(batch), convert_to_numpy=True, show_progress_bar=False)
            batch_vectors = np.asarray(encoded)
            if batch_vectors.ndim != 2:
                raise RuntimeError(f"Unexpected embedding shape from local HF model for {role}: {batch_vectors.shape}")
            if batch_vectors.shape[1] != dimensions:
                raise RuntimeError(
                    f"Embedding dimension mismatch for {role}: expected {dimensions}, got {batch_vectors.shape[1]}"
                )
            if batch_vectors.dtype != np.int8:
                if np.issubdtype(batch_vectors.dtype, np.floating):
                    batch_vectors = np.clip(np.rint(batch_vectors), -128, 127).astype(np.int8)
                else:
                    batch_vectors = batch_vectors.astype(np.int8)
        else:
            if not api_key:
                raise RuntimeError("Missing Perplexity API key. Set PPLX_API_KEY or PERPLEXITY_API_KEY, or run with --mock-embeddings.")
            attempts = 0
            while True:
                attempts += 1
                try:
                    batch_vectors = request_embeddings(batch, api_key, model, dimensions, api_url, timeout_seconds)
                    break
                except urllib.error.HTTPError as exc:
                    body = exc.read().decode("utf-8", errors="ignore")
                    if attempts >= 3:
                        raise RuntimeError(f"Perplexity embeddings request failed for {role} batch {start}:{end}: {body}") from exc
                    time.sleep(2 * attempts)
                except Exception:
                    if attempts >= 3:
                        raise
                    time.sleep(2 * attempts)
        embeddings[start:end] = np.stack(batch_vectors, axis=0)
        embeddings.flush()
        elapsed = max(time.time() - started_at, 1e-6)
        items_done = end - next_index
        rate = items_done / elapsed
        remaining = total - end
        eta_seconds = remaining / rate if rate > 0 else None
        state = {
            "role": role,
            "text_column": text_column,
            "completed": False,
            "next_index": int(end),
            "total": int(total),
            "dimensions": int(dimensions),
            "backend": backend,
            "model": model,
            "updated_at": utc_now(),
        }
        write_json(state_path, state)
        update_progress(
            progress_path,
            {
                "phase": "embedding",
                "role": role,
                "role_status": "running",
                "role_completed": int(end),
                "role_total": int(total),
                "role_percent": round(end / total, 6) if total else 1.0,
                "role_eta_seconds": None if eta_seconds is None else round(eta_seconds, 2),
                "message": f"[{role}] embedded {end}/{total}",
            },
        )
        print(f"[{role}] embedded {end}/{total}")

    final_state = {
        "role": role,
        "text_column": text_column,
        "completed": True,
        "next_index": int(total),
        "total": int(total),
        "dimensions": int(dimensions),
        "backend": backend,
        "model": model,
        "updated_at": utc_now(),
    }
    write_json(state_path, final_state)
    np.savez_compressed(compressed_path, embeddings=np.load(embeddings_path, mmap_mode="r"))
    update_progress(
        progress_path,
        {
            "phase": "embedding",
            "role": role,
            "role_status": "completed",
            "role_completed": int(total),
            "role_total": int(total),
            "role_percent": 1.0,
            "message": f"{role} embeddings completed and compressed to npz.",
        },
    )
    return table, np.load(embeddings_path, mmap_mode="r")


def l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def compute_cosine_features_chunk(
    df: pd.DataFrame,
    query_embeddings: np.ndarray,
    user_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    dims: list[int],
) -> pd.DataFrame:
    result = df.copy()
    query_idx = result["query_text_id"].to_numpy(dtype=np.int32)
    user_idx = result["user_text_id"].to_numpy(dtype=np.int32)
    candidate_idx = result["candidate_text_id"].to_numpy(dtype=np.int32)
    cold_mask = (result["user_profile_text"].fillna("").astype(str) == "coldstart").to_numpy(dtype=bool)
    result["embedding_user_available"] = (~cold_mask).astype(np.int8)

    for dim in dims:
        q = query_embeddings[query_idx, :dim].astype(np.float32, copy=False)
        u = user_embeddings[user_idx, :dim].astype(np.float32, copy=False)
        c = candidate_embeddings[candidate_idx, :dim].astype(np.float32, copy=False)

        qn = l2_normalize_rows(q)
        cn = l2_normalize_rows(c)
        qc_values = (qn * cn).sum(axis=1)

        un = l2_normalize_rows(u)
        uc_values = (un * cn).sum(axis=1)
        qu_values = (qn * un).sum(axis=1)

        combined = qn.copy()
        noncold_idx = ~cold_mask
        if np.any(noncold_idx):
            combined[noncold_idx] = l2_normalize_rows(qn[noncold_idx] + un[noncold_idx])
        quc_values = (combined * cn).sum(axis=1)

        uc_values[cold_mask] = 0.0
        qu_values[cold_mask] = 0.0

        result[f"pplx_qc_cosine_{dim}"] = qc_values
        result[f"pplx_uc_cosine_{dim}"] = uc_values
        result[f"pplx_qu_cosine_{dim}"] = qu_values
        result[f"pplx_quc_cosine_{dim}"] = quc_values

    return result


def persist_feature_split_with_resume(
    output_dir: Path,
    split_name: str,
    df: pd.DataFrame,
    query_embeddings: np.ndarray,
    user_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    dims: list[int],
    chunk_size: int,
    progress_path: Path,
) -> None:
    ensure_dir(output_dir)
    final_path = output_dir / f"{split_name}_with_pplx_features.parquet"
    state_path = output_dir / f"{split_name}_feature_state.json"
    chunk_dir = output_dir / "_chunks" / split_name
    ensure_dir(chunk_dir)

    existing_state = load_json_if_exists(state_path)
    if final_path.exists() and existing_state and existing_state.get("completed"):
        update_progress(
            progress_path,
            {
                "phase": "feature_generation",
                "split": split_name,
                "split_status": "completed",
                "split_completed_rows": int(existing_state.get("next_index", len(df))),
                "split_total_rows": int(len(df)),
                "message": f"{split_name} feature split already available; reusing persisted parquet.",
            },
        )
        return
    if final_path.exists() and not existing_state:
        return

    next_index = int(existing_state.get("next_index", 0)) if existing_state else 0
    started_at = time.time()
    update_progress(
        progress_path,
        {
            "phase": "feature_generation",
            "split": split_name,
            "split_status": "running",
            "split_completed_rows": int(next_index),
            "split_total_rows": int(len(df)),
            "message": f"Building {split_name} feature chunks from row {next_index}.",
        },
    )

    for start in range(next_index, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        chunk_path = chunk_dir / f"{start:09d}_{end:09d}.parquet"
        chunk = df.iloc[start:end].copy()
        chunk = compute_cosine_features_chunk(
            chunk,
            query_embeddings=query_embeddings,
            user_embeddings=user_embeddings,
            candidate_embeddings=candidate_embeddings,
            dims=dims,
        )
        chunk.to_parquet(chunk_path, index=False)
        elapsed = max(time.time() - started_at, 1e-6)
        rows_done = end - next_index
        rate = rows_done / elapsed
        remaining = len(df) - end
        eta_seconds = remaining / rate if rate > 0 else None
        state = {
            "split": split_name,
            "completed": False,
            "next_index": int(end),
            "total_rows": int(len(df)),
            "chunk_size": int(chunk_size),
            "mrl_dims": dims,
            "updated_at": utc_now(),
        }
        write_json(state_path, state)
        update_progress(
            progress_path,
            {
                "phase": "feature_generation",
                "split": split_name,
                "split_status": "running",
                "split_completed_rows": int(end),
                "split_total_rows": int(len(df)),
                "split_percent": round(end / len(df), 6) if len(df) else 1.0,
                "split_eta_seconds": None if eta_seconds is None else round(eta_seconds, 2),
                "message": f"[{split_name}] feature rows {end}/{len(df)}",
            },
        )
        print(f"[{split_name}] feature rows {end}/{len(df)}")

    chunk_paths = sorted(chunk_dir.glob("*.parquet"))
    combined = pd.concat((pd.read_parquet(path) for path in chunk_paths), ignore_index=True)
    combined.to_parquet(final_path, index=False)
    final_state = {
        "split": split_name,
        "completed": True,
        "next_index": int(len(df)),
        "total_rows": int(len(df)),
        "chunk_size": int(chunk_size),
        "mrl_dims": dims,
        "updated_at": utc_now(),
    }
    write_json(state_path, final_state)
    update_progress(
        progress_path,
        {
            "phase": "feature_generation",
            "split": split_name,
            "split_status": "completed",
            "split_completed_rows": int(len(df)),
            "split_total_rows": int(len(df)),
            "split_percent": 1.0,
            "message": f"{split_name} feature split completed.",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Perplexity embedding stores and cosine feature datasets from the behavioral interaction splits.")
    parser.add_argument("--splits-dir", default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-size", choices=sorted(MODEL_SIZE_PRESETS), help="Preset for Perplexity embedding model size.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dimensions", type=positive_int, default=1024)
    parser.add_argument("--mrl-dims", default="256,512,1024")
    parser.add_argument("--batch-size", type=positive_int, default=64, help="Embedding request batch size.")
    parser.add_argument("--chunk-size", type=positive_int, default=10000)
    parser.add_argument("--timeout-seconds", type=positive_int, default=120)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--backend", choices=["api", "hf-local"], default="hf-local")
    parser.add_argument("--hf-model-id", default=DEFAULT_HF_MODEL_ID)
    parser.add_argument("--hf-cache-dir", default=DEFAULT_HF_CACHE_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mock-embeddings", action="store_true")
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    store_dir = output_dir / "store"
    feature_dir = output_dir / "feature_splits"
    progress_path = output_dir / "progress.json"
    ensure_dir(output_dir)
    ensure_dir(store_dir)
    ensure_dir(feature_dir)

    update_progress(
        progress_path,
        {
            "phase": "startup",
            "status": "running",
            "message": "Initializing embedding feature build.",
        },
    )

    api_key = os.getenv("PPLX_API_KEY") or os.getenv("PERPLEXITY_API_KEY")
    mrl_dims = parse_mrl_dims(args.mrl_dims, args.dimensions)
    selected_model, selected_hf_model_id = resolve_model_config(args.model_size, args.model, args.hf_model_id)
    hf_encoder = None
    hf_model_path = None
    resolved_device = resolve_device(args.device)

    if not args.mock_embeddings and args.backend == "hf-local":
        hf_model_path = download_hf_model(selected_hf_model_id, Path(args.hf_cache_dir))
        hf_encoder = load_local_sentence_transformer(hf_model_path, resolved_device)

    frames = load_split_frames(splits_dir)

    role_tables: dict[str, pd.DataFrame] = {}
    role_embeddings: dict[str, np.ndarray] = {}
    for role, column in TEXT_SPECS:
        table = build_unique_text_table(frames, column, role)
        table, embeddings = embed_text_table(
            table=table,
            text_column=column,
            role=role,
            output_dir=store_dir,
            model=selected_model,
            dimensions=args.dimensions,
            batch_size=args.batch_size,
            timeout_seconds=args.timeout_seconds,
            mock_embeddings=args.mock_embeddings,
            api_key=api_key,
            api_url=args.api_url,
            backend=args.backend,
            hf_encoder=hf_encoder,
            progress_path=progress_path,
        )
        role_tables[role] = table
        role_embeddings[role] = embeddings

    mapping_specs = [
        ("query", "query_text", "query_text_id"),
        ("user", "user_profile_text", "user_text_id"),
        ("candidate", "candidate_text", "candidate_text_id"),
    ]
    for split_name, frame in frames.items():
        enriched = frame.copy()
        for role, column, id_column in mapping_specs:
            mapping = role_tables[role][[column, f"{role}_text_id"]].rename(columns={f"{role}_text_id": id_column})
            enriched = enriched.merge(mapping, on=column, how="left", validate="many_to_one")
        persist_feature_split_with_resume(
            output_dir=feature_dir,
            split_name=split_name,
            df=enriched,
            query_embeddings=role_embeddings["query"],
            user_embeddings=role_embeddings["user"],
            candidate_embeddings=role_embeddings["candidate"],
            dims=mrl_dims,
            chunk_size=args.chunk_size,
            progress_path=progress_path,
        )
        print(f"[features] completed {split_name} split with {len(enriched)} rows")

    manifest = {
        "splits_dir": str(splits_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "model": selected_model,
        "model_size": args.model_size or "custom",
        "backend": args.backend,
        "dimensions": args.dimensions,
        "mrl_dims": mrl_dims,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "mock_embeddings": args.mock_embeddings,
        "api_url": args.api_url,
        "hf_model_id": selected_hf_model_id,
        "hf_model_path": hf_model_path,
        "hf_cache_dir": str(Path(args.hf_cache_dir).resolve()),
        "device": resolved_device,
        "text_roles": {
            role: {
                "count": int(len(role_tables[role])),
                "table_path": str((store_dir / f"{role}_texts.parquet").resolve()),
                "embeddings_path": str((store_dir / f"{role}_embeddings.int8.npy").resolve()),
                "compressed_embeddings_path": str((store_dir / f"{role}_embeddings.int8.npz").resolve()),
            }
            for role, _column in TEXT_SPECS
        },
        "feature_files": {
            split_name: str((feature_dir / f"{split_name}_with_pplx_features.parquet").resolve())
            for split_name in SPLIT_NAMES
        },
        "notes": [
            "Embeddings are persisted as raw int8 arrays produced by the selected backend.",
            "Cosine features are computed after per-prefix L2 normalization.",
            "Coldstart recruiter rows set user-based cosine features to zero and use query-only combined cosine.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    update_progress(
        progress_path,
        {
            "phase": "completed",
            "status": "completed",
            "message": "Embedding store and feature splits completed.",
        },
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

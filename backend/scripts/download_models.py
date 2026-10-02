"""Pre-download ML models so the ingestion pipeline never fetches them at runtime.

Usage:
    python backend/scripts/download_models.py

Environment variables (optional):
    NEXUSRAG_EMBEDDING_MODEL  — default: BAAI/bge-m3
    NEXUSRAG_RERANKER_MODEL   — default: BAAI/bge-reranker-v2-m3

Two groups are fetched:

1. sentence-transformers — the bge-m3 vector embedding and the bge-reranker
   cross-encoder, cached under ``~/.cache/huggingface``.
2. Surya (Marker's model stack) — layout/ordering, text detection and OCR-error
   detection, cached under ``~/.cache/huggingface`` (``hf://`` checkpoints) and
   ``~/.cache/datalab`` (``s3://`` checkpoints, served from models.datalab.to).

The Surya group is what makes the Docker image self-contained. Marker spawns its
model servers on first parse and waits 300s for each to report healthy; on a slow
link the download loses that race and the parse fails outright rather than just
being slow. Note the ``s3://`` checkpoints are *not* on HuggingFace, so an
``HF_ENDPOINT`` mirror does nothing for them.
"""
import os
import time


def download_sentence_transformers() -> None:
    embedding_model = os.environ.get("NEXUSRAG_EMBEDDING_MODEL", "BAAI/bge-m3")
    reranker_model = os.environ.get("NEXUSRAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

    from sentence_transformers import SentenceTransformer, CrossEncoder

    print(f"[st] embedding model: {embedding_model}")
    SentenceTransformer(embedding_model)

    print(f"[st] reranker model: {reranker_model}")
    CrossEncoder(reranker_model)


def _download_s3_checkpoint(checkpoint: str) -> None:
    """Fetch an ``s3://`` checkpoint from models.datalab.to into the datalab cache."""
    from surya.common.s3 import S3DownloaderMixin, download_directory

    local_dir = S3DownloaderMixin.get_local_path(checkpoint)
    remote_path = checkpoint[len(S3DownloaderMixin.s3_prefix):]

    # download_directory is not itself resilient; without retries one dropped
    # connection fails the whole image build.
    for attempt in range(1, 4):
        try:
            download_directory(remote_path, local_dir)
            print(f"[surya] {remote_path} -> {local_dir}")
            return
        except Exception as e:  # noqa: BLE001 - retried below
            print(f"[surya] {remote_path} attempt {attempt}/3 failed: {e}")
            if attempt == 3:
                raise
            time.sleep(5)


def _download_hf_checkpoint(checkpoint: str) -> None:
    """Fetch an ``hf://<repo>[/<subfolder>]`` checkpoint through the HF cache."""
    from huggingface_hub import snapshot_download

    # A repo id is itself "org/name", so only segments past that are subfolders:
    # hf://datalab-to/surya_layout2/order -> repo "datalab-to/surya_layout2",
    # subfolder "order".
    parts = checkpoint[len("hf://"):].split("/")
    repo_id = "/".join(parts[:2])
    subfolder = "/".join(parts[2:])
    snapshot_download(
        repo_id,
        allow_patterns=[f"{subfolder}/**"] if subfolder else None,
    )
    print(f"[surya] {checkpoint} -> huggingface cache ({repo_id})")


def download_surya_models() -> None:
    """Pre-fetch every Surya checkpoint the document parsers reference."""
    from surya.settings import settings as surya_settings

    # Read from surya's own settings rather than hardcoding, so a Surya upgrade
    # that repoints a checkpoint does not silently leave it unfetched.
    checkpoints = [
        surya_settings.FAST_LAYOUT_MODEL_CHECKPOINT,
        surya_settings.FAST_ORDER_MODEL_CHECKPOINT,
        surya_settings.DETECTOR_MODEL_CHECKPOINT,
        surya_settings.OCR_ERROR_MODEL_CHECKPOINT,
    ]

    for checkpoint in dict.fromkeys(c for c in checkpoints if c):
        if checkpoint.startswith("s3://"):
            _download_s3_checkpoint(checkpoint)
        elif checkpoint.startswith("hf://"):
            _download_hf_checkpoint(checkpoint)
        else:
            print(f"[surya] skip {checkpoint!r} (local path, nothing to fetch)")


def download_models() -> None:
    download_sentence_transformers()
    download_surya_models()

    print("\nAll models downloaded successfully.")


if __name__ == "__main__":
    download_models()

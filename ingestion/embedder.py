"""
Embedding Module - Local Only.

Runs entirely offline using sentence-transformers.
No API key required. No network calls after first model download.

Stable configuration for macOS (no SIGKILL) and Proxmox LXC (no GPU):
- Always runs on CPU by default
- Small batch size to avoid memory spikes
- Inter-batch sleep to prevent sustained 100% CPU

Model options (configure via LOCAL_EMBEDDING_MODEL in .env):
  BAAI/bge-small-en-v1.5  → 384 dims, ~33MB   ← fastest, lowest RAM
  BAAI/bge-base-en-v1.5   → 768 dims, ~110MB  ← default, good balance
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Common interface for all embedding backends."""

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    def embed_one(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]


class LocalEmbedder(BaseEmbedder):
    """
    Local CPU-based embedding using sentence-transformers.

    Designed to be stable on:
    - macOS (avoids MPS/Metal SIGKILL by forcing CPU)
    - Proxmox LXC (no GPU available)
    - Low-RAM machines (small batch, inter-batch sleep)

    First run downloads the model from HuggingFace (~33MB or ~110MB).
    Subsequent runs are fully offline.

    Configuration via .env:
        LOCAL_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5  # or bge-base-en-v1.5
        LOCAL_EMBEDDING_BATCH_SIZE=8                   # lower = less CPU spike
        LOCAL_EMBEDDING_SLEEP=0.1                      # seconds between batches
    """

    _DIM_MAP = {
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
    }

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 8,
        inter_batch_sleep: float = 0.05,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.inter_batch_sleep = inter_batch_sleep
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "\n[ERROR] sentence-transformers is not installed.\n"
                "Run: pip install sentence-transformers\n"
            )
        print(f"  [Embedder] Loading model: {self.model_name} (device=cpu)")
        t0 = time.time()
        # Force CPU — avoids macOS MPS SIGKILL and works in Proxmox LXC
        self._model = SentenceTransformer(self.model_name, device="cpu")
        print(f"  [Embedder] Ready in {time.time() - t0:.1f}s | dim={self.dimension}")

    @property
    def dimension(self) -> int:
        return self._DIM_MAP.get(self.model_name, 768)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            chunk = [t[:4000] for t in texts[i: i + self.batch_size]]
            vecs = self._model.encode(
                chunk,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).tolist()
            all_vectors.extend(vecs)

            # Small sleep between batches to avoid sustained 100% CPU
            if self.inter_batch_sleep > 0 and i + self.batch_size < len(texts):
                time.sleep(self.inter_batch_sleep)

        return all_vectors


def create_embedder() -> LocalEmbedder:
    """
    Create a LocalEmbedder from environment variables.

    Environment variables:
        LOCAL_EMBEDDING_MODEL       BAAI/bge-small-en-v1.5 (default, 384 dims)
                                    BAAI/bge-base-en-v1.5  (768 dims, more accurate)
        LOCAL_EMBEDDING_BATCH_SIZE  Number of texts per batch (default: 8)
        LOCAL_EMBEDDING_SLEEP       Sleep seconds between batches (default: 0.05)
    """
    model = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    batch_size = int(os.getenv("LOCAL_EMBEDDING_BATCH_SIZE", "8"))
    sleep = float(os.getenv("LOCAL_EMBEDDING_SLEEP", "0.05"))

    print(f"  [Embedder] model={model} | batch={batch_size} | sleep={sleep}s")
    return LocalEmbedder(
        model_name=model,
        batch_size=batch_size,
        inter_batch_sleep=sleep,
    )

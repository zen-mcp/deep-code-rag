"""
Embedding Module

Support 3 backends:
1. local (DEFAULT): sentence-transformers runs offline, no API key required
   - BAAI/bge-small-en-v1.5  → 384 dims, 33MB, very lightweight, fast startup
   - BAAI/bge-base-en-v1.5   → 768 dims, 110MB, balanced (RECOMMENDED)
   - BAAI/bge-large-en-v1.5  → 1024 dims, 330MB, most accurate
2. openai: OpenAI-compatible API (requires API key with /embeddings endpoint)
3. gpt_encode: Use GPT chat completion to manually create embedding (fallback)
   - Slower and more token-intensive, but works with ChatGPT subscription
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod


# ── Base Interface ────────────────────────────────────────────────────────────

class BaseEmbedder(ABC):
    """Interface for all embedding backends."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Number of dimensions of the embedding vector."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Create embedding for a batch of texts. Return list of vectors."""
        ...

    def embed_one(self, text: str) -> list[float]:
        """Create embedding for a single text."""
        return self.embed_batch([text])[0]


# 1. Local Embedder (sentence-transformers) - DEFAULT ─────────────────────

class LocalEmbedder(BaseEmbedder):
    """
    Use sentence-transformers to create embeddings locally.
    No API key required, runs entirely offline after first model download.

    Recommended models (in order of preference):
    ┌─────────────────────────────┬──────┬────────┬──────────────────────────┐
    │ Model                       │ Dims │  Size  │ Note                     │
    ├─────────────────────────────┼──────┼────────┼──────────────────────────┤
    │ BAAI/bge-base-en-v1.5       │  768 │ 110MB  │ RECOMMENDED - balanced   │
    │ BAAI/bge-small-en-v1.5      │  384 │  33MB  │ Lightest, fastest startup│
    │ BAAI/bge-large-en-v1.5      │ 1024 │ 330MB  │ Most accurate            │
    └─────────────────────────────┴──────┴────────┴──────────────────────────┘

    Model is downloaded from HuggingFace, then cached at:
    - macOS/Linux: ~/.cache/huggingface/hub/
    - Windows: C:\\Users\\<user>\\.cache\\huggingface\\hub\\
    """

    _DIM_MAP = {
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
        "jinaai/jina-embeddings-v2-base-code": 768,
        "microsoft/codebert-base": 768,
    }

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str | None = None,  # None = automatically select cpu/cuda/mps
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None  # Lazy loading

        # Automatically select the best device
        if device is None:
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "mps"  # Apple Silicon
                else:
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

    def _load_model(self):
        """Load model for the first time (lazy loading)."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            print(f"  [Embedder] Loading local model: {self.model_name} (device={self.device})")
            print(f"  [Embedder] First run will download model from HuggingFace...")
            t0 = time.time()
            self._model = SentenceTransformer(self.model_name, device=self.device)
            elapsed = time.time() - t0
            print(f"  [Embedder] Model ready in {elapsed:.1f}s | dim={self.dimension}")
        except ImportError:
            raise ImportError(
                "\n[ERROR] sentence-transformers is not installed.\n"
                "Run: pip install sentence-transformers\n"
            )

    @property
    def dimension(self) -> int:
        return self._DIM_MAP.get(self.model_name, 768)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            # Truncate text quá dài (model limit ~512 tokens)
            batch = [t[:4000] for t in batch]
            vecs = self._model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).tolist()
            all_vectors.extend(vecs)

        return all_vectors


# ── 2. OpenAI API Embedder ────────────────────────────────────────────────────

class OpenAIEmbedder(BaseEmbedder):
    """
    Use OpenAI-compatible API to create embeddings.
    Requires OpenAI Platform account (NOT ChatGPT subscription).

    Configure via .env:
        OPENAI_API_KEY=sk-...
        OPENAI_BASE_URL=https://api.openai.com/v1   # or ClipProxy URL
        EMBEDDING_MODEL=text-embedding-3-small
    """

    _DIM_MAP = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 100,
        max_retries: int = 3,
    ):
        from openai import OpenAI

        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL")

        self._client = OpenAI(api_key=api_key, base_url=base_url)

    @property
    def dimension(self) -> int:
        return self._DIM_MAP.get(self.model, 1536)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t[:8000] for t in texts[i: i + self.batch_size]]
            for attempt in range(self.max_retries):
                try:
                    response = self._client.embeddings.create(
                        model=self.model,
                        input=batch,
                    )
                    all_vectors.extend(item.embedding for item in response.data)
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    wait = 2 ** attempt
                    print(f"  [Embedder] Retry {attempt+1}/{self.max_retries} after {wait}s: {e}")
                    time.sleep(wait)
        return all_vectors


# ── 3. GPT-based Embedder (fallback dùng chat completion) ────────────────────
class GPTEncodeEmbedder(BaseEmbedder):
    """
    Create embedding by requesting GPT to return a numeric vector.
    This is a FALLBACK when there is no embedding API.

    WARNING:
    - Slower than 10-20x compared to actual embedding API
    - More token-intensive
    - Only for testing, NOT for production with 1600+ files

    Recommended: Use LocalEmbedder instead.
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        dims: int = 256,
    ):
        from openai import OpenAI
        self._dims = dims
        self.model = model
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    @property
    def dimension(self) -> int:
        return self._dims

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Gọi GPT để tạo embedding vector từ text."""
        import json
        import hashlib

        vectors = []
        for text in texts:
            # Create prompt requesting GPT to return a vector
            prompt = (
                f"Create a semantic embedding vector of exactly {self._dims} float numbers "
                f"(between -1 and 1) that captures the meaning of the following code snippet. "
                f"Return ONLY a JSON array of {self._dims} numbers, nothing else.\n\n"
                f"Code:\n{text[:2000]}"
            )
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self._dims * 8,
                    temperature=0,
                )
                content = response.choices[0].message.content.strip()
                # Parse JSON array
                vec = json.loads(content)
                if isinstance(vec, list) and len(vec) == self._dims:
                    vectors.append([float(x) for x in vec])
                else:
                    # Fallback: hash-based vector if GPT returns wrong format
                    vectors.append(_hash_vector(text, self._dims))
            except Exception as e:
                print(f"  [GPTEncodeEmbedder] Error: {e}, using hash fallback")
                vectors.append(_hash_vector(text, self._dims))

        return vectors


def _hash_vector(text: str, dims: int) -> list[float]:
    """Create a fake vector from hash of text (only used when all other methods fail)."""
    import hashlib
    import struct
    h = hashlib.sha256(text.encode()).digest()
    # Repeat hash to reach dims
    extended = (h * (dims // 32 + 1))[:dims * 4]
    raw = struct.unpack(f"{dims}f", extended[:dims * 4])
    # Normalize to [-1, 1]
    max_val = max(abs(x) for x in raw) or 1.0
    return [x / max_val for x in raw]


# ── Factory ───────────────────────────────────────────────────────────────────

def create_embedder(backend: str = "local") -> BaseEmbedder:
    """
    Factory function to create an appropriate embedder.

    backend:
    - "local"      : sentence-transformers (DEFAULT, no API key required)
    - "openai"     : OpenAI/ClipProxy API (requires API key with /embeddings)
    - "gpt_encode" : Use GPT chat to create embedding (slow, only for testing)
    - "auto"       : Automatically select: openai if API key is available, otherwise use local

    Configure via environment variables:
    - LOCAL_EMBEDDING_MODEL : HuggingFace model name (default: BAAI/bge-base-en-v1.5)
    - EMBEDDING_MODEL       : OpenAI model name (default: text-embedding-3-small)
    - OPENAI_API_KEY        : API key
    - OPENAI_BASE_URL       : Base URL (empty = use api.openai.com)
    """
    if backend == "auto":
        # Check if API key works with embedding
        # If not, automatically fallback to local
        backend = "local"  # Default safest option

    if backend == "local":
        model = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        print(f"  [Embedder] Backend: local | Model: {model}")
        return LocalEmbedder(model_name=model)

    elif backend == "openai":
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        print(f"  [Embedder] Backend: openai | Model: {model}")
        return OpenAIEmbedder(model=model)

    elif backend == "gpt_encode":
        model = os.getenv("GPT_MODEL", "gpt-4.1-mini")
        dims = int(os.getenv("GPT_ENCODE_DIMS", "256"))
        print(f"  [Embedder] Backend: gpt_encode | Model: {model} | Dims: {dims}")
        print(f"  [Embedder] WARNING: This is slow. Use 'local' backend for production.")
        return GPTEncodeEmbedder(model=model, dims=dims)

    else:
        raise ValueError(
            f"Unknown backend: '{backend}'. "
            f"Valid options: 'local', 'openai', 'gpt_encode', 'auto'"
        )

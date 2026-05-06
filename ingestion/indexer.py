"""
Qdrant Indexer.

This module combines Embedder and Qdrant Client to:
1. Create collection in Qdrant (if not exists)
2. Upsert EnrichedChunk (embed text -> vector -> upsert to Qdrant)
3. Support incremental indexing (only update changed chunks)
4. Provide CLI to run full indexing
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from ingestion.chunker import EnrichedChunk, iter_project_chunks
from ingestion.embedder import BaseEmbedder, create_embedder


# ── Config ────────────────────────────────────────────────────────────────────

COLLECTION_NAME = "codebase"   # Tên collection trong Qdrant (dùng chung cho mọi project)
EMBED_BATCH_SIZE = 50          # Số chunk embed cùng lúc


# ── Indexer ───────────────────────────────────────────────────────────────────

class QdrantIndexer:
    """
    Combine Embedder + Qdrant to index the entire codebase.
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: str | None = None,
        embedder: BaseEmbedder | None = None,
        collection_name: str = COLLECTION_NAME,
    ):
        self.collection_name = collection_name

        # Initialize Qdrant client
        self.qdrant = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=30,
        )

        # Initialize embedder (lazy: only create when needed)
        self._embedder = embedder
        self._embedder_initialized = embedder is not None

    @property
    def embedder(self) -> BaseEmbedder:
        if not self._embedder_initialized:
            self._embedder = create_embedder(
                backend=os.getenv("EMBEDDING_BACKEND", "local")
            )
            self._embedder_initialized = True
        return self._embedder

    def ensure_collection(self) -> None:
        """Create Qdrant collection if not exists (or recreate if dimension changes)."""
        existing = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection_name in existing:
            # Check if dimension matches (important when changing model)
            info = self.qdrant.get_collection(self.collection_name)
            existing_dim = info.config.params.vectors.size
            if existing_dim != self.embedder.dimension:
                print(f"  [Qdrant] WARNING: Dimension mismatch! "
                      f"Collection has {existing_dim} dims, "
                      f"embedder produces {self.embedder.dimension} dims.")
                print(f"  [Qdrant] Deleting and recreating collection...")
                self.qdrant.delete_collection(self.collection_name)
            else:
                print(f"  [Qdrant] Collection '{self.collection_name}' OK (dim={existing_dim}).")
                return

        print(f"  [Qdrant] Creating collection '{self.collection_name}' "
              f"(dim={self.embedder.dimension})...")
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedder.dimension,
                distance=Distance.COSINE,
            ),
        )

        # Create payload index to filter quickly by project_id, symbol_type, ...
        for field in ["project_id", "symbol_type", "angular_pattern", "file_path"]:
            self.qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema="keyword",
            )
        print(f"  [Qdrant] Collection created with payload indexes.")

    def index_project(
        self,
        project_id: str,
        repo_root: Path,
        incremental: bool = True,
    ) -> dict:
        """
        Index toàn bộ một project vào Qdrant.

        Args:
            project_id: Project ID (e.g. "web-blogic-view")
            repo_root: Path to repository root
            incremental: If True, only update changed chunks

        Returns:
            Dict statistics: { indexed, skipped, total_time }
        """
        self.ensure_collection()

        start_time = time.time()
        total_indexed = 0
        total_skipped = 0
        batch_num = 0

        print(f"\n[Indexer] Starting indexing project: {project_id}")
        print(f"[Indexer] Repo root: {repo_root}")
        print(f"[Indexer] Mode: {'incremental' if incremental else 'full'}")
        print("-" * 60)

        for batch in iter_project_chunks(project_id, repo_root, batch_size=50):
            batch_num += 1

            if incremental:
                # Filter out chunks that haven't changed
                batch, skipped = self._filter_unchanged(batch)
                total_skipped += skipped

            if not batch:
                continue

            # Create embeddings for batch
            texts = [ec.embed_text for ec in batch]
            try:
                vectors = self.embedder.embed_batch(texts)
            except Exception as e:
                print(f"  [Embedder] ERROR in batch {batch_num}: {e}")
                continue

            # Upsert to Qdrant
            points = [
                PointStruct(
                    id=_chunk_id_to_int(ec.chunk_id),
                    vector=vector,
                    payload=ec.payload,
                )
                for ec, vector in zip(batch, vectors)
            ]

            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

            total_indexed += len(batch)
            elapsed = time.time() - start_time
            print(
                f"  Batch {batch_num:3d}: +{len(batch):3d} indexed "
                f"| {total_indexed:5d} total "
                f"| {elapsed:.1f}s elapsed"
            )

        total_time = time.time() - start_time
        print("-" * 60)
        print(f"[Indexer] Done! Indexed: {total_indexed}, Skipped: {total_skipped}")
        print(f"[Indexer] Total time: {total_time:.1f}s")

        return {
            "project_id": project_id,
            "indexed": total_indexed,
            "skipped": total_skipped,
            "total_time": total_time,
        }

    def _filter_unchanged(
        self, batch: list[EnrichedChunk]
    ) -> tuple[list[EnrichedChunk], int]:
        """
        Check if chunk exists in Qdrant (compare content_hash).
        Return (chunks_to_update, num_skipped).
        """
        if not batch:
            return [], 0

        # Get IDs to check
        ids = [_chunk_id_to_int(ec.chunk_id) for ec in batch]

        try:
            existing = self.qdrant.retrieve(
                collection_name=self.collection_name,
                ids=ids,
                with_payload=["content_hash"],
            )
        except Exception:
            return batch, 0

        # Create map: id -> content_hash
        existing_hashes = {
            str(p.id): p.payload.get("content_hash", "")
            for p in existing
        }

        to_update: list[EnrichedChunk] = []
        skipped = 0
        for ec in batch:
            chunk_int_id = str(_chunk_id_to_int(ec.chunk_id))
            current_hash = ec.payload.get("content_hash", "")
            if existing_hashes.get(chunk_int_id) == current_hash:
                skipped += 1
            else:
                to_update.append(ec)

        return to_update, skipped

    def delete_project(self, project_id: str) -> int:
        """Delete all data of a project from Qdrant."""
        result = self.qdrant.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="project_id",
                        match=MatchValue(value=project_id),
                    )
                ]
            ),
        )
        print(f"[Indexer] Deleted all data for project: {project_id}")
        return 0

    def get_stats(self, project_id: str | None = None) -> dict:
        """Get statistics about data in Qdrant."""
        info = self.qdrant.get_collection(self.collection_name)
        stats = {
            "collection": self.collection_name,
            "total_points": info.points_count,
            "vectors_count": info.vectors_count,
        }

        if project_id:
            # Count chunks of specific project
            count_result = self.qdrant.count(
                collection_name=self.collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="project_id",
                            match=MatchValue(value=project_id),
                        )
                    ]
                ),
            )
            stats["project_points"] = count_result.count

        return stats


# ── Helper ────────────────────────────────────────────────────────────────────

def _chunk_id_to_int(chunk_id: str) -> int:
    """Convert hex chunk_id to integer (Qdrant uses int or UUID as ID)."""
    return int(chunk_id[:16], 16)


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    """
    CLI to run indexing from command line.
    Example:
        python -m ingestion.indexer --project web-blogic-view --repo /path/to/repo
    """
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Index a codebase into Qdrant")
    parser.add_argument("--project", required=True, help="Project ID (e.g. web-blogic-view)")
    parser.add_argument("--repo", required=True, help="Path to repository root")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--full", action="store_true", help="Full re-index (ignore cache)")
    parser.add_argument("--delete", action="store_true", help="Delete project data first")
    args = parser.parse_args()

    repo_root = Path(args.repo)
    if not repo_root.exists():
        print(f"ERROR: Repo not found: {repo_root}")
        return

    indexer = QdrantIndexer(qdrant_url=args.qdrant_url)

    if args.delete:
        indexer.delete_project(args.project)

    result = indexer.index_project(
        project_id=args.project,
        repo_root=repo_root,
        incremental=not args.full,
    )
    print("\nResult:", result)


if __name__ == "__main__":
    main()

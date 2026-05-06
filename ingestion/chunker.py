"""
Semantic Chunker and Metadata Enrichment.

This module processes CodeChunks from AST Parser:
1. Split large chunks (> MAX_TOKENS) by method boundaries
2. Enrich metadata (read agents/*.md, attach project rules)
3. Create final text for embedding
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from ingestion.ast_parser import CodeChunk

# Configuration
MAX_CHUNK_CHARS = 6000    # ~1500 tokens
OVERLAP_CHARS = 300       # Overlap between sub-chunks to avoid losing context


@dataclass
class EnrichedChunk:
    """
    Enriched chunk ready for embedding and saving to Qdrant.
    """
    # Content for embedding (content + context header)
    embed_text: str

    # Full metadata for Qdrant payload
    payload: dict

    # Unique ID (used for upsert to Qdrant)
    chunk_id: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            import hashlib
            key = f"{self.payload['file_path']}:{self.payload['symbol_name']}:{self.payload['start_line']}"
            self.chunk_id = hashlib.sha256(key.encode()).hexdigest()[:16]


# Project Context Loader

class ProjectContextLoader:
    """
    Read .agents/*.md files to create project context.
    This context is embedded into the header of each chunk to let LLM
    always know which project and rules it is working on.
    """

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self._cache: dict[str, str] = {}

    def load_agents_context(self) -> str:
        """Read and summarize content from .agents/*.md."""
        if "agents_context" in self._cache:
            return self._cache["agents_context"]

        agents_dir = self.repo_root / ".agents"
        if not agents_dir.exists():
            return ""

        # Priority order to read
        priority_files = ["CONTEXT.md", "ARCHITECTURE.md", "RULES.md"]
        parts: list[str] = []

        for fname in priority_files:
            fpath = agents_dir / fname
            if fpath.exists():
                content = fpath.read_text(encoding="utf-8", errors="replace")
                # Only take first 2000 characters of each file to avoid too long
                parts.append(f"### {fname}\n{content[:2000]}")

        context = "\n\n".join(parts)
        self._cache["agents_context"] = context
        return context

    def get_short_context(self) -> str:
        """Summarize project context (used for chunk header)."""
        if "short_context" in self._cache:
            return self._cache["short_context"]

        context_file = self.repo_root / ".agents" / "CONTEXT.md"
        if not context_file.exists():
            return ""

        content = context_file.read_text(encoding="utf-8", errors="replace")
        # Take first part (usually the overview)
        lines = content.splitlines()
        short = "\n".join(lines[:20])
        self._cache["short_context"] = short
        return short


# Text Splitter for large chunk

def _split_by_methods(content: str, max_chars: int, overlap: int) -> list[str]:
    """
    Split a large code segment by method/function boundaries.
    Prefer to cut at lines starting with method signature.
    """
    if len(content) <= max_chars:
        return [content]

    # Find possible cut points (method signatures, blank lines)
    method_pattern = re.compile(
        r"^\s{2,}(?:async\s+)?(?:private|public|protected|readonly)?\s*"
        r"(?:async\s+)?\w+\s*\(",
        re.MULTILINE,
    )
    cut_points = [0]
    for match in method_pattern.finditer(content):
        cut_points.append(match.start())
    cut_points.append(len(content))

    chunks: list[str] = []
    current_start = 0

    for i in range(1, len(cut_points)):
        segment_end = cut_points[i]
        segment = content[current_start:segment_end]

        if len(segment) > max_chars and i > 1:
            # Segment too large, cut at previous cut_point
            prev_cut = cut_points[i - 1]
            chunk_text = content[current_start:prev_cut]
            if chunk_text.strip():
                chunks.append(chunk_text)
            current_start = max(0, prev_cut - overlap)

    # Remaining part
    remaining = content[current_start:]
    if remaining.strip():
        chunks.append(remaining)

    # Fallback: if still too large, split by character count
    result: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            result.append(chunk)
        else:
            # Split by character count, cut at nearest newline
            start = 0
            while start < len(chunk):
                end = start + max_chars
                if end < len(chunk):
                    # Find nearest newline to cut nicely
                    newline_pos = chunk.rfind("\n", start, end)
                    if newline_pos > start:
                        end = newline_pos
                result.append(chunk[start:end])
                start = end - overlap

    return result or [content]


# Main Chunker

class SemanticChunker:
    """
    Process list of CodeChunk from AST Parser:
    - Split large chunks
    - Create embed_text with context header
    - Create full metadata for Qdrant
    """

    def __init__(self, project_id: str, repo_root: Path):
        self.project_id = project_id
        self.repo_root = repo_root
        self.ctx_loader = ProjectContextLoader(repo_root)

    def process(self, chunks: list[CodeChunk]) -> list[EnrichedChunk]:
        """Process list of CodeChunk, return list of EnrichedChunk."""
        enriched: list[EnrichedChunk] = []
        for chunk in chunks:
            enriched.extend(self._enrich(chunk))
        return enriched

    def _enrich(self, chunk: CodeChunk) -> list[EnrichedChunk]:
        """Enrich a CodeChunk, may create multiple EnrichedChunk if too large."""
        # Split if needed
        sub_contents = _split_by_methods(
            chunk.content, MAX_CHUNK_CHARS, OVERLAP_CHARS
        )

        results: list[EnrichedChunk] = []
        for idx, sub_content in enumerate(sub_contents):
            embed_text = self._build_embed_text(chunk, sub_content)
            payload = self._build_payload(chunk, sub_content, idx, len(sub_contents))
            results.append(EnrichedChunk(embed_text=embed_text, payload=payload))

        return results

    def _build_embed_text(self, chunk: CodeChunk, content: str) -> str:
        """
        Create text for embedding.
        Add context header to let model understand the context of the code segment.
        """
        lines: list[str] = []

        # Header: information about project and file
        lines.append(f"Project: {self.project_id}")
        lines.append(f"File: {chunk.file_path}")

        if chunk.symbol_type and chunk.symbol_name:
            lines.append(f"Symbol: {chunk.symbol_type} {chunk.symbol_name}")

        if chunk.angular_pattern:
            lines.append(f"Pattern: {chunk.angular_pattern}")

        if chunk.selector:
            lines.append(f"Selector: {chunk.selector}")

        if chunk.dependencies:
            lines.append(f"Injects: {', '.join(chunk.dependencies)}")

        if chunk.decorators:
            lines.append(f"Decorators: {', '.join(chunk.decorators)}")

        lines.append("")  # Blank line
        lines.append(content)

        return "\n".join(lines)

    def _build_payload(
        self, chunk: CodeChunk, content: str, sub_idx: int, total_subs: int
    ) -> dict:
        """Create metadata payload for saving to Qdrant."""
        return {
            # Identification
            "project_id": chunk.project_id,
            "file_path": chunk.file_path,

            # Symbol info
            "symbol_type": chunk.symbol_type,
            "symbol_name": chunk.symbol_name,
            "angular_pattern": chunk.angular_pattern,
            "selector": chunk.selector,

            # Position
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,

            # Relationships
            "dependencies": chunk.dependencies,
            "decorators": chunk.decorators,
            "exported": chunk.exported,

            # Content
            "content": content,
            "has_template": bool(chunk.template_content),

            # Sub-chunk info (if split)
            "sub_index": sub_idx,
            "total_subs": total_subs,

            # For incremental indexing
            "content_hash": _hash_content(content),
        }


def _hash_content(content: str) -> str:
    import hashlib
    return hashlib.md5(content.encode()).hexdigest()


# Iterator for the entire project

def iter_project_chunks(
    project_id: str,
    repo_root: Path,
    batch_size: int = 50,
) -> Iterator[list[EnrichedChunk]]:
    """
    Generator: iterate through the entire project and yield each batch of EnrichedChunk.
    Used for indexing with progress tracking.
    """
    from ingestion.ast_parser import AngularASTParser, discover_files

    parser = AngularASTParser(project_id=project_id, repo_root=repo_root)
    chunker = SemanticChunker(project_id=project_id, repo_root=repo_root)

    files = discover_files(repo_root)
    batch: list[EnrichedChunk] = []

    for file_path in files:
        raw_chunks = parser.parse_file(file_path)
        enriched = chunker.process(raw_chunks)
        batch.extend(enriched)

        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch

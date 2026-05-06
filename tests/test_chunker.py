"""
Test Semantic Chunker with real project web-blogic-view.
Run: python tests/test_chunker.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.ast_parser import AngularASTParser, discover_files
from ingestion.chunker import SemanticChunker, iter_project_chunks


def test_chunker():
    repo_root = Path("/home/ubuntu/web-blogic-view")
    project_id = "web-blogic-view"

    parser = AngularASTParser(project_id=project_id, repo_root=repo_root)
    chunker = SemanticChunker(project_id=project_id, repo_root=repo_root)

    # Test 1: View embed_text of a component
    print("=" * 60)
    print("TEST 1: Embed text của một Smart Component")
    files = discover_files(repo_root)
    component_files = [
        f for f in files
        if "component" in f.name and f.suffix == ".ts"
        and f.with_suffix(".html").exists()
        and "spec" not in f.name
    ]
    if component_files:
        test_file = component_files[0]
        raw_chunks = parser.parse_file(test_file)
        enriched = chunker.process(raw_chunks)
        for ec in enriched:
            print(f"\nChunk ID: {ec.chunk_id}")
            print(f"Symbol: {ec.payload['symbol_type']} / {ec.payload['symbol_name']}")
            print(f"Pattern: {ec.payload['angular_pattern']}")
            print(f"Dependencies: {ec.payload['dependencies']}")
            print(f"Has template: {ec.payload['has_template']}")
            print(f"Embed text length: {len(ec.embed_text)} chars")
            print("\n--- Embed Text Preview (first 500 chars) ---")
            print(ec.embed_text[:500])
            print("...")

    # Test 2: Stats for the entire project (first 100 files)
    print("\n" + "=" * 60)
    print("TEST 2: Stats cho 100 file đầu")
    all_files = discover_files(repo_root)[:100]
    total_chunks = 0
    total_chars = 0
    pattern_stats: dict[str, int] = {}

    for f in all_files:
        raw = parser.parse_file(f)
        enriched = chunker.process(raw)
        total_chunks += len(enriched)
        for ec in enriched:
            total_chars += len(ec.embed_text)
            p = ec.payload.get("angular_pattern") or ec.payload.get("symbol_type", "unknown")
            pattern_stats[p] = pattern_stats.get(p, 0) + 1

    print(f"  Total enriched chunks: {total_chunks}")
    print(f"  Total embed chars:     {total_chars:,}")
    print(f"  Avg chars per chunk:   {total_chars // max(total_chunks, 1):,}")
    print("\n  Pattern breakdown:")
    for k, v in sorted(pattern_stats.items(), key=lambda x: -x[1]):
        bar = "#" * (v * 30 // max(list(pattern_stats.values()) + [1]))
        print(f"    {k:35s} {v:4d}  {bar}")

    print("\n✅ Chunker tests passed!")


if __name__ == "__main__":
    test_chunker()

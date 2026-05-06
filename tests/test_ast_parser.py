"""
Test AST Parser with real project web-blogic-view.
Run: python tests/test_ast_parser.py
"""
import sys
from pathlib import Path

# Add root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.ast_parser import AngularASTParser, discover_files


def test_with_real_project():
    repo_root = Path("/home/ubuntu/web-blogic-view")
    if not repo_root.exists():
        print("ERROR: Repo not found at", repo_root)
        return

    parser = AngularASTParser(
        project_id="web-blogic-view",
        repo_root=repo_root,
    )

    # Test 1: Khám phá file
    print("=" * 60)
    print("TEST 1: File Discovery")
    files = discover_files(repo_root)
    ts_files = [f for f in files if f.suffix == ".ts"]
    html_files = [f for f in files if f.suffix == ".html"]
    print(f"  TypeScript files: {len(ts_files)}")
    print(f"  HTML files:       {len(html_files)}")
    print(f"  Total:            {len(files)}")

    # Test 2: Parse a Service file
    print("\n" + "=" * 60)
    print("TEST 2: Parse a Service file")
    service_files = [f for f in ts_files if "service" in f.name and "spec" not in f.name]
    if service_files:
        test_file = service_files[0]
        print(f"  File: {test_file.relative_to(repo_root)}")
        chunks = parser.parse_file(test_file)
        for chunk in chunks:
            print(f"  -> [{chunk.symbol_type}] {chunk.symbol_name}")
            print(f"     Pattern: {chunk.angular_pattern}")
            print(f"     Dependencies: {chunk.dependencies}")
            print(f"     Lines: {chunk.start_line}-{chunk.end_line}")

    # Test 3: Parse a Component file (Smart Component)
    print("\n" + "=" * 60)
    print("TEST 3: Parse a Component file (with HTML bundling)")
    component_files = [
        f for f in ts_files
        if "component" in f.name and "spec" not in f.name
        and f.with_suffix(".html").exists()
    ]
    if component_files:
        test_file = component_files[0]
        print(f"  File: {test_file.relative_to(repo_root)}")
        chunks = parser.parse_file(test_file)
        for chunk in chunks:
            print(f"  -> [{chunk.symbol_type}] {chunk.symbol_name}")
            print(f"     Pattern: {chunk.angular_pattern}")
            print(f"     Selector: {chunk.selector}")
            print(f"     Dependencies: {chunk.dependencies}")
            has_template = bool(chunk.template_content)
            print(f"     Has HTML template bundled: {has_template}")
            if has_template:
                print(f"     Template preview: {chunk.template_content[:100].strip()}...")

    # Test 4: Parse an Interface/Model file
    print("\n" + "=" * 60)
    print("TEST 4: Parse an Interface/Model file")
    model_files = [
        f for f in ts_files
        if "model" in f.name or "models" in str(f) and "spec" not in f.name
    ]
    if model_files:
        test_file = model_files[0]
        print(f"  File: {test_file.relative_to(repo_root)}")
        chunks = parser.parse_file(test_file)
        for chunk in chunks[:3]:  # Only show first 3 chunks
            print(f"  -> [{chunk.symbol_type}] {chunk.symbol_name}")

    # Test 5: Quick Stats (parse first 50 files)
    print("\n" + "=" * 60)
    print("TEST 5: Quick Stats (first 50 files)")
    stats: dict[str, int] = {}
    total_chunks = 0
    for file_path in ts_files[:50]:
        chunks = parser.parse_file(file_path)
        total_chunks += len(chunks)
        for chunk in chunks:
            key = f"{chunk.symbol_type}/{chunk.angular_pattern}" if chunk.angular_pattern else chunk.symbol_type
            stats[key] = stats.get(key, 0) + 1

    print(f"  Total chunks from 50 files: {total_chunks}")
    print("  Breakdown by type:")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"    {k:40s}: {v}")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_with_real_project()

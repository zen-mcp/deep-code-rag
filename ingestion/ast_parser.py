"""
Angular AST Parser using Tree-sitter.

Parse Angular TypeScript and HTML files,
extract entities (class, function, interface, decorator)
and create rich metadata for each chunk.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tree_sitter_html as tshtml
import tree_sitter_typescript as tsts
from tree_sitter import Language, Node, Parser

# Initialize Language and Parser
TS_LANGUAGE = Language(tsts.language_typescript())
TSX_LANGUAGE = Language(tsts.language_tsx())
HTML_LANGUAGE = Language(tshtml.language())

_ts_parser = Parser(TS_LANGUAGE)
_html_parser = Parser(HTML_LANGUAGE)


# Data Models
SymbolType = Literal[
    "class", "interface", "function", "method",
    "component", "service", "directive", "pipe",
    "module", "guard", "interceptor", "unknown",
]


@dataclass
class CodeChunk:
    """Represent a code chunk with semantic information extracted from AST."""

    # Content
    content: str
    file_path: str
    project_id: str

    # Classification
    symbol_type: SymbolType = "unknown"
    symbol_name: str = ""

    # Position in file
    start_line: int = 0
    end_line: int = 0

    # Angular Metadata
    angular_pattern: str = ""          # "smart-component", "dumb-component", "service", ...
    dependencies: list[str] = field(default_factory=list)   # Services injected
    template_content: str = ""         # HTML template content (if any)
    selector: str = ""                 # Angular selector (e.g.: "app-login")

    # Additional Metadata
    exported: bool = False
    decorators: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "file_path": self.file_path,
            "project_id": self.project_id,
            "symbol_type": self.symbol_type,
            "symbol_name": self.symbol_name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "angular_pattern": self.angular_pattern,
            "dependencies": self.dependencies,
            "template_content": self.template_content,
            "selector": self.selector,
            "exported": self.exported,
            "decorators": self.decorators,
        }


# Helper Functions

def _node_text(node: Node, source: bytes) -> str:
    """Get text of a node from source code."""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _find_nodes(node: Node, node_type: str) -> list[Node]:
    """Find all nodes with specific type in the subtree."""
    results = []
    if node.type == node_type:
        results.append(node)
    for child in node.children:
        results.extend(_find_nodes(child, node_type))
    return results


def _find_nodes_multi(node: Node, node_types: set[str]) -> list[Node]:
    """Find all nodes with type in the set."""
    results = []
    if node.type in node_types:
        results.append(node)
    for child in node.children:
        results.extend(_find_nodes_multi(child, node_types))
    return results


def _extract_decorator_names(class_node: Node, source: bytes) -> list[str]:
    """Extract decorator names of a class (e.g.: Component, Injectable)."""
    decorators = []
    # Decorator is at the same level as class_declaration, before it
    parent = class_node.parent
    if parent is None:
        return decorators
    for child in parent.children:
        if child.end_byte <= class_node.start_byte and child.type == "decorator":
            # Get decorator name: @Component -> "Component"
            for sub in child.children:
                if sub.type == "call_expression":
                    func = sub.child_by_field_name("function")
                    if func:
                        decorators.append(_node_text(func, source))
                elif sub.type == "identifier":
                    decorators.append(_node_text(sub, source))
    return decorators


def _extract_angular_metadata(class_node: Node, source: bytes) -> dict:
    """
    Extract Angular metadata from decorator @Component / @Injectable.
    Return: { selector, template_url, style_urls, provided_in }
    """
    meta = {"selector": "", "template_url": "", "style_urls": [], "provided_in": ""}
    parent = class_node.parent
    if parent is None:
        return meta

    for child in parent.children:
        if child.end_byte > class_node.start_byte:
            break
        if child.type != "decorator":
            continue
        text = _node_text(child, source)

        # Extract selector
        sel_match = re.search(r"selector\s*:\s*['\"]([^'\"]+)['\"]", text)
        if sel_match:
            meta["selector"] = sel_match.group(1)

        # Extract templateUrl
        tpl_match = re.search(r"templateUrl\s*:\s*['\"]([^'\"]+)['\"]", text)
        if tpl_match:
            meta["template_url"] = tpl_match.group(1)

    return meta


def _extract_injected_dependencies(class_node: Node, source: bytes) -> list[str]:
    """
    Extract services injected into class through:
    - inject(ServiceName)
    - constructor(private svc: ServiceName)
    """
    deps = set()
    text = _node_text(class_node, source)

    # Pattern 1: inject(ServiceName)
    for match in re.finditer(r"\binject\s*\(\s*([A-Z][A-Za-z0-9_]*)", text):
        deps.add(match.group(1))

    # Pattern 2: constructor(private/readonly name: ServiceType)
    for match in re.finditer(
        r"(?:private|public|protected|readonly)\s+\w+\s*:\s*([A-Z][A-Za-z0-9_]*)",
        text,
    ):
        type_name = match.group(1)
        # Filter out primitive types
        if type_name not in {"String", "Number", "Boolean", "Array", "Object"}:
            deps.add(type_name)

    return sorted(deps)


def _classify_angular_pattern(
    class_name: str,
    decorators: list[str],
    dependencies: list[str],
) -> tuple[SymbolType, str]:
    """
    Classify Angular pattern of a class.
    Return: (symbol_type, angular_pattern)
    """
    if "Component" in decorators:
        # Smart component: has injected service
        # Dumb component: only uses input()/output()
        pattern = "smart-component" if dependencies else "dumb-component"
        return "component", pattern

    if "Injectable" in decorators:
        if class_name.endswith("Guard") or "Guard" in decorators:
            return "guard", "guard"
        if class_name.endswith("Interceptor"):
            return "interceptor", "interceptor"
        return "service", "service"

    if "Directive" in decorators:
        return "directive", "directive"

    if "Pipe" in decorators:
        return "pipe", "pipe"

    if "NgModule" in decorators:
        return "module", "module"

    return "class", "class"


# Main Parser Class

class AngularASTParser:
    """
    Main parser to parse Angular/TypeScript files.

    Use Tree-sitter to build AST and extract chunks
    with semantic information (class, interface, function) and rich metadata.
    """

    def __init__(self, project_id: str, repo_root: Path):
        self.project_id = project_id
        self.repo_root = repo_root

    def parse_file(self, file_path: Path) -> list[CodeChunk]:
        """Parse a file and return a list of CodeChunk."""
        suffix = file_path.suffix.lower()

        if suffix in (".ts", ".tsx"):
            return self._parse_typescript(file_path)
        elif suffix in (".html",):
            return self._parse_html(file_path)
        else:
            return []

    def _parse_typescript(self, file_path: Path) -> list[CodeChunk]:
        """Parse file TypeScript/Angular, extract class and interface."""
        try:
            source = file_path.read_bytes()
        except (OSError, PermissionError):
            return []

        tree = _ts_parser.parse(source)
        chunks: list[CodeChunk] = []

        # Find all class declarations
        class_nodes = _find_nodes(tree.root_node, "class_declaration")
        for class_node in class_nodes:
            chunk = self._extract_class_chunk(class_node, source, file_path)
            if chunk:
                # If it is a Component, find and merge template HTML
                if chunk.symbol_type == "component":
                    self._attach_html_template(chunk, file_path)
                chunks.append(chunk)

        # Find all interface declarations
        iface_nodes = _find_nodes(tree.root_node, "interface_declaration")
        for iface_node in iface_nodes:
            chunk = self._extract_interface_chunk(iface_node, source, file_path)
            if chunk:
                chunks.append(chunk)

        # If file has no class/interface (e.g.: file helper/utils), get the whole file
        if not chunks:
            content = source.decode("utf-8", errors="replace")
            if content.strip():
                rel_path = str(file_path.relative_to(self.repo_root))
                chunks.append(CodeChunk(
                    content=content,
                    file_path=rel_path,
                    project_id=self.project_id,
                    symbol_type="unknown",
                    symbol_name=file_path.stem,
                    start_line=1,
                    end_line=len(content.splitlines()),
                ))

        return chunks

    def _extract_class_chunk(
        self, class_node: Node, source: bytes, file_path: Path
    ) -> CodeChunk | None:
        """Extract information from a class declaration node."""
        # Get class name
        name_node = class_node.child_by_field_name("name")
        if not name_node:
            return None
        class_name = _node_text(name_node, source)

        # Extract decorators
        decorators = _extract_decorator_names(class_node, source)

        # Extract Angular metadata (selector, templateUrl)
        ang_meta = _extract_angular_metadata(class_node, source)

        # Extract dependencies (inject calls)
        dependencies = _extract_injected_dependencies(class_node, source)

        # Classify pattern
        symbol_type, angular_pattern = _classify_angular_pattern(
            class_name, decorators, dependencies
        )

        # Get all text of class (including decorator above)
        # Find first decorator before class
        content_start = class_node.start_byte
        parent = class_node.parent
        if parent:
            for child in parent.children:
                if child.end_byte <= class_node.start_byte and child.type == "decorator":
                    content_start = min(content_start, child.start_byte)

        content = source[content_start:class_node.end_byte].decode("utf-8", errors="replace")
        rel_path = str(file_path.relative_to(self.repo_root))

        # Check exported
        exported = False
        if class_node.parent:
            for sib in class_node.parent.children:
                if sib.type == "export_statement" and class_node in _find_nodes(sib, "class_declaration"):
                    exported = True
                    break
        # Simplified: check text
        if "export class" in content or "export default class" in content:
            exported = True

        return CodeChunk(
            content=content,
            file_path=rel_path,
            project_id=self.project_id,
            symbol_type=symbol_type,
            symbol_name=class_name,
            start_line=class_node.start_point[0] + 1,
            end_line=class_node.end_point[0] + 1,
            angular_pattern=angular_pattern,
            dependencies=dependencies,
            selector=ang_meta["selector"],
            exported=exported,
            decorators=decorators,
        )

    def _extract_interface_chunk(
        self, iface_node: Node, source: bytes, file_path: Path
    ) -> CodeChunk | None:
        """Extract information from a interface declaration node."""
        name_node = iface_node.child_by_field_name("name")
        if not name_node:
            return None
        iface_name = _node_text(name_node, source)
        content = _node_text(iface_node, source)
        rel_path = str(file_path.relative_to(self.repo_root))

        return CodeChunk(
            content=content,
            file_path=rel_path,
            project_id=self.project_id,
            symbol_type="interface",
            symbol_name=iface_name,
            start_line=iface_node.start_point[0] + 1,
            end_line=iface_node.end_point[0] + 1,
            exported="export interface" in content,
        )

    def _attach_html_template(self, chunk: CodeChunk, ts_file: Path) -> None:
        """
        Find and merge HTML template into Component chunk.
        Support both inline template and external templateUrl.
        """
        # Find HTML file with the same name (e.g.: sign-in.component.ts -> sign-in.component.html)
        html_file = ts_file.with_suffix(".html")
        if html_file.exists():
            try:
                template_content = html_file.read_text(encoding="utf-8", errors="replace")
                chunk.template_content = template_content
                # Merge into content to have enough context for embedding
                chunk.content = (
                    f"// TypeScript Component\n{chunk.content}\n\n"
                    f"// HTML Template ({html_file.name})\n{template_content}"
                )
            except OSError:
                pass

    def _parse_html(self, file_path: Path) -> list[CodeChunk]:
        """
        Parse file HTML standalone (not template of component).
        Only process if there is no .ts file with the same name (to avoid duplicate).
        """
        ts_counterpart = file_path.with_suffix(".ts")
        if ts_counterpart.exists():
            # This template will be processed by _attach_html_template
            return []

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []

        rel_path = str(file_path.relative_to(self.repo_root))
        return [CodeChunk(
            content=content,
            file_path=rel_path,
            project_id=self.project_id,
            symbol_type="unknown",
            symbol_name=file_path.stem,
            start_line=1,
            end_line=len(content.splitlines()),
        )]


# File Discovery

EXCLUDED_DIRS = {
    "node_modules", "dist", "build", ".git", ".cache",
    "coverage", ".angular", "tmp", "deprecated",
}

SUPPORTED_EXTENSIONS = {".ts", ".html"}


def discover_files(repo_root: Path) -> list[Path]:
    """
    Discover all TypeScript and HTML files in the repository,
    skipping unnecessary directories.
    """
    files: list[Path] = []
    for path in repo_root.rglob("*"):
        # Skip excluded directories
        if any(excluded in path.parts for excluded in EXCLUDED_DIRS):
            continue
        # Skip test and spec files
        if path.stem.endswith((".spec", ".test", ".d")):
            continue
        if path.suffix in SUPPORTED_EXTENSIONS and path.is_file():
            files.append(path)
    return sorted(files)

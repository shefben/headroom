"""Graph/symbol-scoped code context (#11) — the ``CodeGraphAdapter`` owner.

Reuses Headroom's existing code-aware infrastructure instead of adding a new
parser or downloader:

* **tree-sitter** through the compression stack's thread-local parser cache
  (:func:`headroom.transforms.code_compressor._get_parser`);
* the **ast-grep** interceptor's language table and runner
  (:mod:`headroom.proxy.interceptors.astgrep`), used only when tree-sitter
  cannot parse a supported language and the binary is already present
  (:func:`headroom.binaries.which` never downloads);
* the code-graph watcher's source-extension and ignore tables
  (:mod:`headroom.graph.watcher`).

Unsupported languages fall back to exact path and lexical identifiers.

Model: nodes ``file / module / class / function / symbol``; edges ``imports /
includes / calls / references / inherits / contains / defines``. Each trusted
worktree root has its own graph with per-file content hashes, so updates are
incremental: a changed file is re-parsed alone (``invalidate_file`` from
``FILE_CHANGE`` events, plus a rate-limited stat-only rescan). Ignored,
generated and vendored paths are skipped; ``git ls-files --exclude-standard``
honours ``.gitignore`` when available.

All indexing runs on one background thread. Request-path queries
(:meth:`CodeGraphAdapter.neighbors` etc.) read only the in-memory graph under
a short lock — no SQLite, subprocess or file I/O on the event loop. Per-file
records persist in the agent ledger so a restart only re-parses changed files.

Graph ranking is one retrieval feature: callers must never use it to omit
user-named evidence or required diff context.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from headroom.log_safety import describe_exception

if TYPE_CHECKING:
    from headroom.agent.identity import WorkspaceIdentity
    from headroom.agent.ledger_db import AgentLedgerDB

log = logging.getLogger(__name__)

DEFAULT_RADIUS = 2
MAX_FILES = 10_000
MAX_FILE_BYTES = 512 * 1024
MAX_IDENTIFIERS = 1500
MAX_WORKSPACES = 8
RESCAN_INTERVAL_S = 60.0
MAX_VISITED = 400

_GENERATED_SEGMENTS = frozenset(
    {
        "vendor",
        "vendors",
        "third_party",
        "third-party",
        "thirdparty",
        "generated",
        "__generated__",
        "gen",
        "node_modules",
        "bower_components",
        "site-packages",
        "dist",
        "build",
        "out",
        "target",
        "coverage",
    }
)
_GENERATED_SUFFIXES = (
    ".min.js",
    ".min.css",
    "_pb2.py",
    "_pb2_grpc.py",
    ".pb.go",
    ".pb.cc",
    ".pb.h",
    ".generated.cs",
    ".g.dart",
    ".bundle.js",
    ".lock",
)

# ast-grep language name -> tree-sitter-language-pack name
_TS_LANG = {
    "python": "python",
    "typescript": "typescript",
    "tsx": "tsx",
    "javascript": "javascript",
    "jsx": "javascript",
    "go": "go",
    "rust": "rust",
    "java": "java",
    "ruby": "ruby",
    "c": "c",
    "cpp": "cpp",
}

_DEF_TYPES: dict[str, dict[str, str]] = {
    "python": {"function_definition": "function", "class_definition": "class"},
    "javascript": {
        "function_declaration": "function",
        "generator_function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "function",
    },
    "typescript": {
        "function_declaration": "function",
        "class_declaration": "class",
        "abstract_class_declaration": "class",
        "interface_declaration": "class",
        "method_definition": "function",
        "type_alias_declaration": "symbol",
        "enum_declaration": "symbol",
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "function",
        "type_spec": "class",
    },
    "rust": {
        "function_item": "function",
        "struct_item": "class",
        "enum_item": "class",
        "trait_item": "class",
        "mod_item": "module",
    },
    "java": {
        "class_declaration": "class",
        "interface_declaration": "class",
        "enum_declaration": "class",
        "method_declaration": "function",
        "constructor_declaration": "function",
    },
    "ruby": {
        "method": "function",
        "singleton_method": "function",
        "class": "class",
        "module": "module",
    },
    "c": {"function_definition": "function", "struct_specifier": "class"},
    "cpp": {
        "function_definition": "function",
        "struct_specifier": "class",
        "class_specifier": "class",
        "namespace_definition": "module",
    },
}
_DEF_TYPES["tsx"] = _DEF_TYPES["typescript"]
_CALL_TYPES = frozenset({"call", "call_expression", "method_invocation", "new_expression"})
_IMPORT_TYPES = frozenset(
    {
        "import_statement",
        "import_from_statement",
        "import_declaration",
        "use_declaration",
        "import_spec",
    }
)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SYMBOLISH = re.compile(r"^(?=.*(?:[A-Z]|_))[A-Za-z_][A-Za-z0-9_]{2,}$|^[a-z][a-z0-9]{3,}$")
_LEX_DEF = re.compile(
    r"^\s*(?:export\s+|pub(?:\([^)]*\))?\s+|public\s+|private\s+|static\s+|async\s+)*"
    r"(def|class|function|func|fn|sub|module|interface|struct|enum|trait|object)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)
_LEX_IMPORT = re.compile(
    r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.\"'/@-]+)|"
    r"#\s*include\s*[<\"]([^>\"]+)[>\"]|use\s+([\w:]+)|require(?:_relative)?\s*\(?\s*[\"']([^\"']+))",
    re.MULTILINE,
)


def _source_extensions() -> frozenset[str]:
    from headroom.graph.watcher import _SOURCE_EXTENSIONS

    return frozenset(e.lower() for e in _SOURCE_EXTENSIONS)


def _ignore_dirs() -> frozenset[str]:
    from headroom.graph.watcher import _IGNORE_DIRS

    return frozenset(_IGNORE_DIRS)


def _ext_to_lang() -> dict[str, str]:
    from headroom.proxy.interceptors.astgrep import _EXT_TO_LANG

    return dict(_EXT_TO_LANG)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Definition:
    kind: str  # class | function | module | symbol
    qualname: str
    line: int
    parent: str | None = None


@dataclass
class FileRecord:
    path: str
    hash: str
    mtime_ns: int
    size: int
    lang: str
    parser: str  # tree-sitter | ast-grep | lexical
    defs: list[Definition] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    calls: list[tuple[str, str]] = field(default_factory=list)  # (caller qualname | "", callee)
    bases: list[tuple[str, str]] = field(default_factory=list)  # (class qualname, base name)
    identifiers: list[str] = field(default_factory=list)

    def payload(self) -> str:
        return json.dumps(
            {
                "defs": [[d.kind, d.qualname, d.line, d.parent] for d in self.defs],
                "imports": self.imports,
                "calls": self.calls,
                "bases": self.bases,
                "identifiers": self.identifiers,
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_row(
        cls, path: str, hash_: str, mtime_ns: int, size: int, lang: str, parser: str, payload: str
    ) -> FileRecord:
        data = json.loads(payload)
        return cls(
            path=path,
            hash=hash_,
            mtime_ns=int(mtime_ns),
            size=int(size),
            lang=lang,
            parser=parser,
            defs=[Definition(k, q, int(ln), p) for k, q, ln, p in data.get("defs") or []],
            imports=list(data.get("imports") or []),
            calls=[(str(a), str(b)) for a, b in data.get("calls") or []],
            bases=[(str(a), str(b)) for a, b in data.get("bases") or []],
            identifiers=list(data.get("identifiers") or []),
        )


@dataclass(frozen=True)
class GraphNode:
    id: str
    kind: str  # file | module | class | function | symbol
    name: str
    path: str | None
    line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "path": self.path,
            "line": self.line,
        }


@dataclass(frozen=True)
class Neighbor:
    node: GraphNode
    distance: int
    relation: str  # exact_symbol | exact_path | same_file | <edge kind> | lexical
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.node.to_dict(),
            "distance": self.distance,
            "relation": self.relation,
            "score": round(self.score, 4),
        }

    @property
    def label(self) -> str:
        if self.node.kind == "file":
            return self.node.path or self.node.name
        return f"{self.node.name} ({self.node.path}:{self.node.line})"


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _node_text(node: Any) -> str:
    try:
        return node.text.decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - grammar nodes differ across versions
        return ""


def _def_name(node: Any) -> str | None:
    name = node.child_by_field_name("name")
    if name is not None:
        return _node_text(name)
    if node.type == "impl_item":
        typ = node.child_by_field_name("type")
        return _node_text(typ) if typ is not None else None
    declarator = node.child_by_field_name("declarator")
    depth = 0
    while declarator is not None and depth < 6:
        if declarator.type in (
            "identifier",
            "field_identifier",
            "qualified_identifier",
            "destructor_name",
            "operator_name",
        ):
            return _node_text(declarator).split("::")[-1]
        nxt = declarator.child_by_field_name("declarator")
        if nxt is None:
            for child in declarator.children:
                if child.type in ("identifier", "field_identifier", "qualified_identifier"):
                    return _node_text(child).split("::")[-1]
        declarator = nxt
        depth += 1
    return None


def _last_identifier(text: str) -> str | None:
    found = _IDENT.findall(text)
    return found[-1] if found else None


def _import_names(node: Any, lang: str) -> list[str]:
    text = _node_text(node)
    if lang == "python":
        names = re.findall(r"(?:from\s+([\w.]+))|(?:import\s+([\w., ]+))", text)
        out: list[str] = []
        for frm, imp in names:
            if frm:
                out.append(frm)
            elif imp:
                out.extend(p.strip().split(" ")[0] for p in imp.split(",") if p.strip())
        return out
    strings = re.findall(r"[\"']([^\"']+)[\"']", text)
    if strings:
        return strings
    if lang == "rust":
        return [text.removeprefix("use").strip().rstrip(";").split("::{")[0]]
    return re.findall(r"[\w.]+", text)[1:2]


def _ts_extract(source: bytes, lang: str, record: FileRecord) -> bool:
    from headroom.transforms.code_compressor import _get_parser

    ts_lang = _TS_LANG.get(lang)
    def_types = _DEF_TYPES.get(ts_lang or "")
    if not ts_lang or def_types is None:
        return False
    try:
        tree = _get_parser(ts_lang).parse(source)
    except (ImportError, ValueError):
        return False
    stack: list[tuple[Any, tuple[str, ...]]] = [(tree.root_node, ())]
    while stack:
        node, scope = stack.pop()
        ntype = node.type
        new_scope = scope
        if ntype in def_types:
            name = _def_name(node)
            if name:
                qual = ".".join((*scope, name))
                record.defs.append(
                    Definition(
                        def_types[ntype],
                        qual,
                        node.start_point[0] + 1,
                        ".".join(scope) if scope else None,
                    )
                )
                new_scope = (*scope, name)
                if def_types[ntype] == "class":
                    for field_name in (
                        "superclasses",
                        "superclass",
                        "super_interfaces",
                        "base_class_clause",
                    ):
                        sup = node.child_by_field_name(field_name)
                        if sup is not None:
                            for base in _IDENT.findall(_node_text(sup)):
                                if base not in ("extends", "implements", "public", "private"):
                                    record.bases.append((qual, base))
                    for child in node.children:
                        if child.type in ("class_heritage", "base_class_clause", "superclass"):
                            for base in _IDENT.findall(_node_text(child)):
                                if base not in ("extends", "implements", "public", "private"):
                                    record.bases.append((qual, base))
        elif ntype == "impl_item":
            typ = node.child_by_field_name("type")
            if typ is not None:
                new_scope = (*scope, _node_text(typ))
        elif ntype in _IMPORT_TYPES or ntype == "preproc_include":
            if ntype == "preproc_include":
                path = node.child_by_field_name("path")
                if path is not None:
                    record.imports.append(_node_text(path).strip('"<>'))
            else:
                record.imports.extend(_import_names(node, ts_lang))
            continue
        elif ntype in _CALL_TYPES:
            target = (
                node.child_by_field_name("function")
                or node.child_by_field_name("name")
                or node.child_by_field_name("method")
                or node.child_by_field_name("constructor")
            )
            callee = _last_identifier(_node_text(target)) if target is not None else None
            if callee and ntype == "method_invocation":
                callee = _node_text(node.child_by_field_name("name") or target)
            if callee:
                record.calls.append((".".join(scope), callee))
            if ts_lang in ("javascript", "typescript", "tsx") and callee == "require":
                args = node.child_by_field_name("arguments")
                if args is not None:
                    record.imports.extend(re.findall(r"[\"']([^\"']+)[\"']", _node_text(args)))
        for child in reversed(node.children):
            stack.append((child, new_scope))
    record.parser = "tree-sitter"
    return True


def _astgrep_extract(source: str, lang: str, record: FileRecord) -> bool:
    from headroom import binaries
    from headroom.proxy.interceptors.astgrep import _run_ast_grep

    exe = binaries.which("ast-grep")
    if exe is None:
        return False
    matches = _run_ast_grep(exe, lang, source)
    if not matches:
        return False
    for match in matches:
        meta = match.get("metaVariables", {}).get("single", {})
        name = (meta.get("NAME") or {}).get("text")
        line = match.get("range", {}).get("start", {}).get("line")
        if not name or not isinstance(line, int):
            continue
        head = str(match.get("text", "")).lstrip().split(" ", 1)[0]
        kind = "class" if head in ("class", "struct", "enum", "interface") else "function"
        record.defs.append(Definition(kind, name, line + 1))
    record.parser = "ast-grep"
    return True


def _lexical_extract(source: str, record: FileRecord) -> None:
    for match in _LEX_DEF.finditer(source):
        keyword, name = match.group(1), match.group(2)
        kind = {
            "class": "class",
            "struct": "class",
            "interface": "class",
            "trait": "class",
            "enum": "class",
            "object": "class",
            "module": "module",
        }.get(keyword, "function")
        record.defs.append(Definition(kind, name, source.count("\n", 0, match.start()) + 1))
    for match in _LEX_IMPORT.finditer(source):
        value = next((g for g in match.groups() if g), None)
        if value:
            record.imports.append(value.strip("\"'"))
    record.parser = "lexical"


def extract_file(path: str, source_bytes: bytes, *, lang: str | None) -> FileRecord:
    """Parse one file into a :class:`FileRecord` (tree-sitter → ast-grep → lexical)."""
    record = FileRecord(
        path=path,
        hash=hashlib.sha256(source_bytes).hexdigest(),
        mtime_ns=0,
        size=len(source_bytes),
        lang=lang or "text",
        parser="lexical",
    )
    text = source_bytes.decode("utf-8", "replace")
    parsed = False
    if lang:
        try:
            parsed = _ts_extract(source_bytes, lang, record)
        except Exception as exc:  # noqa: BLE001 - a grammar failure falls back
            log.debug("tree-sitter graph extraction failed for %s: %s", lang, exc)
            record.defs.clear()
            record.imports.clear()
            record.calls.clear()
            record.bases.clear()
        if not parsed:
            try:
                parsed = _astgrep_extract(text, lang, record)
            except Exception as exc:  # noqa: BLE001
                log.debug("ast-grep graph extraction failed: %s", exc)
    if not parsed:
        _lexical_extract(text, record)
    seen: dict[str, None] = {}
    for ident in _IDENT.findall(text):
        if _SYMBOLISH.match(ident) and ident not in seen:
            seen[ident] = None
            if len(seen) >= MAX_IDENTIFIERS:
                break
    record.identifiers = list(seen)
    record.calls = list(dict.fromkeys(record.calls))[:4000]
    record.imports = list(dict.fromkeys(i for i in record.imports if i))[:400]
    return record


# ---------------------------------------------------------------------------
# Per-workspace graph
# ---------------------------------------------------------------------------


def _module_names(path: str) -> list[str]:
    stem = path.rsplit(".", 1)[0]
    parts = [p for p in stem.split("/") if p]
    if parts and parts[-1] in ("__init__", "index", "mod", "lib"):
        parts = parts[:-1]
    names = []
    if parts:
        names.append(".".join(parts))
        names.append("/".join(parts))
        names.append(parts[-1])
        for i in range(1, len(parts)):
            names.append(".".join(parts[i:]))
    return names


class WorkspaceGraph:
    """In-memory graph for one trusted worktree root."""

    def __init__(self, workspace_id: str, root: str) -> None:
        self.workspace_id = workspace_id
        self.root = root
        self.lock = threading.RLock()
        self.files: dict[str, FileRecord] = {}
        self.all_paths: set[str] = set()
        self.lower_paths: dict[str, str] = {}
        self.defs_by_name: dict[str, set[tuple[str, str]]] = {}
        self.module_index: dict[str, set[str]] = {}
        self.importers: dict[str, set[str]] = {}
        self.callers: dict[str, set[tuple[str, str]]] = {}
        self.subclasses: dict[str, set[tuple[str, str]]] = {}
        self.referencers: dict[str, set[str]] = {}
        self.ready = False
        self.indexing = False
        self.last_scan = 0.0
        self.parse_counts: dict[str, int] = {"tree-sitter": 0, "ast-grep": 0, "lexical": 0}
        self.dirty: set[str] = set()

    # -- maintenance (indexer thread) --------------------------------------

    def _unindex(self, record: FileRecord) -> None:
        for d in record.defs:
            name = d.qualname.rsplit(".", 1)[-1]
            bucket = self.defs_by_name.get(name)
            if bucket:
                bucket.discard((record.path, d.qualname))
                if not bucket:
                    self.defs_by_name.pop(name, None)
        for name in _module_names(record.path):
            bucket = self.module_index.get(name)
            if bucket:
                bucket.discard(record.path)
        for caller, callee in record.calls:
            bucket = self.callers.get(callee)
            if bucket:
                bucket.discard((record.path, caller))
        for cls, base in record.bases:
            bucket = self.subclasses.get(base)
            if bucket:
                bucket.discard((record.path, cls))
        for ident in record.identifiers:
            bucket = self.referencers.get(ident)
            if bucket:
                bucket.discard(record.path)
        self._rebuild_importers_for(record.path, remove=True)

    def _index(self, record: FileRecord) -> None:
        for d in record.defs:
            self.defs_by_name.setdefault(d.qualname.rsplit(".", 1)[-1], set()).add(
                (record.path, d.qualname)
            )
        for name in _module_names(record.path):
            self.module_index.setdefault(name, set()).add(record.path)
        for caller, callee in record.calls:
            self.callers.setdefault(callee, set()).add((record.path, caller))
        for cls, base in record.bases:
            self.subclasses.setdefault(base, set()).add((record.path, cls))
        for ident in record.identifiers:
            self.referencers.setdefault(ident, set()).add(record.path)

    def _rebuild_importers_for(self, path: str, *, remove: bool) -> None:
        record = self.files.get(path)
        if record is None:
            return
        for target in self.resolve_imports(record):
            bucket = self.importers.setdefault(target, set())
            if remove:
                bucket.discard(path)
            else:
                bucket.add(path)

    def put(self, record: FileRecord) -> None:
        with self.lock:
            old = self.files.get(record.path)
            if old is not None:
                self._unindex(old)
            self.files[record.path] = record
            self._add_path(record.path)
            self._index(record)
            self._rebuild_importers_for(record.path, remove=False)

    def remove(self, path: str) -> None:
        with self.lock:
            old = self.files.get(path)
            if old is not None:
                self._unindex(old)
                del self.files[path]
            self.all_paths.discard(path)
            self.lower_paths.pop(path.lower(), None)

    def _add_path(self, path: str) -> None:
        self.all_paths.add(path)
        self.lower_paths[path.lower()] = path

    def resolve_imports(self, record: FileRecord) -> set[str]:
        out: set[str] = set()
        base_dir = record.path.rsplit("/", 1)[0] if "/" in record.path else ""
        for name in record.imports:
            candidates: set[str] = set()
            cleaned = name.strip()
            if cleaned.startswith("."):
                rel = os.path.normpath(os.path.join(base_dir, cleaned)).replace("\\", "/")
                for path in self.all_paths:
                    if path.rsplit(".", 1)[0] in (rel, rel + "/index", rel + "/__init__"):
                        candidates.add(path)
                if not candidates:
                    dotted = cleaned.lstrip(".").replace("/", ".")
                    candidates |= self.module_index.get(dotted, set())
            else:
                key = cleaned.replace("::", ".").removeprefix("crate.").removeprefix("self.")
                candidates |= self.module_index.get(key, set())
                if not candidates and "/" in cleaned:
                    candidates |= self.module_index.get(cleaned.rsplit(".", 1)[0], set())
                if not candidates and "." in key:
                    candidates |= self.module_index.get(key.rsplit(".", 1)[0], set())
            out |= {c for c in candidates if c != record.path}
        return out

    def normalize_path(self, value: str) -> str | None:
        """Map a resource id / relative / absolute path to a known relative path."""
        text = value.strip().strip("`'\"")
        if text.startswith("path:"):
            text = text[5:]
        text = text.replace("\\", "/")
        root = self.root.replace("\\", "/").rstrip("/")
        lowered = text.lower()
        if lowered.startswith(root.lower() + "/"):
            text = text[len(root) + 1 :]
        text = re.sub(r"^(?:\./)+", "", text)
        if text in self.all_paths:
            return text
        found = self.lower_paths.get(text.lower())
        if found:
            return found
        if "/" not in text and text:
            matches = [p for p in self.all_paths if p.rsplit("/", 1)[-1] == text]
            if len(matches) == 1:
                return matches[0]
        return None

    # -- queries (request path; lock held briefly) -------------------------

    def _file_node(self, path: str) -> GraphNode:
        return GraphNode(f"file:{path}", "file", path.rsplit("/", 1)[-1], path)

    def _def_node(self, path: str, qualname: str) -> GraphNode | None:
        record = self.files.get(path)
        if record is None:
            return None
        for d in record.defs:
            if d.qualname == qualname:
                return GraphNode(f"sym:{path}#{qualname}", d.kind, qualname, path, d.line)
        return None

    def adjacent(self, node: GraphNode) -> list[tuple[str, GraphNode]]:
        out: list[tuple[str, GraphNode]] = []
        if node.kind == "file" and node.path:
            record = self.files.get(node.path)
            if record is not None:
                for d in record.defs:
                    if d.parent is None:
                        out.append(
                            (
                                "defines",
                                GraphNode(
                                    f"sym:{node.path}#{d.qualname}",
                                    d.kind,
                                    d.qualname,
                                    node.path,
                                    d.line,
                                ),
                            )
                        )
                for target in sorted(self.resolve_imports(record)):
                    edge = "includes" if record.lang in ("c", "cpp") else "imports"
                    out.append((edge, self._file_node(target)))
            for importer in sorted(self.importers.get(node.path, ())):
                out.append(("imported_by", self._file_node(importer)))
            return out
        if node.path is None:
            return out
        record = self.files.get(node.path)
        if record is None:
            return out
        qual = node.name
        short = qual.rsplit(".", 1)[-1]
        out.append(("defined_in", self._file_node(node.path)))
        parent_qual = qual.rsplit(".", 1)[0] if "." in qual else None
        for d in record.defs:
            if d.parent == qual:
                out.append(
                    (
                        "contains",
                        GraphNode(
                            f"sym:{node.path}#{d.qualname}", d.kind, d.qualname, node.path, d.line
                        ),
                    )
                )
            elif parent_qual is not None and d.qualname == parent_qual:
                out.append(
                    (
                        "contained_by",
                        GraphNode(
                            f"sym:{node.path}#{d.qualname}", d.kind, d.qualname, node.path, d.line
                        ),
                    )
                )
        for path, caller in sorted(self.callers.get(short, ())):
            target = self._def_node(path, caller) if caller else None
            out.append(("called_by", target or self._file_node(path)))
        for caller, callee in record.calls:
            if caller == qual:
                for path, q in sorted(self.defs_by_name.get(callee, ())):
                    target = self._def_node(path, q)
                    if target is not None:
                        out.append(("calls", target))
        for cls, base in record.bases:
            if cls == qual:
                for path, q in sorted(self.defs_by_name.get(base, ())):
                    target = self._def_node(path, q)
                    if target is not None:
                        out.append(("inherits", target))
        for path, cls in sorted(self.subclasses.get(short, ())):
            target = self._def_node(path, cls)
            if target is not None:
                out.append(("inherited_by", target))
        for path in sorted(self.referencers.get(short, ())):
            if path != node.path:
                out.append(("referenced_by", self._file_node(path)))
        return out

    def resolve_targets(self, target: str) -> list[tuple[GraphNode, str]]:
        path = self.normalize_path(target)
        if path is not None:
            return [(self._file_node(path), "exact_path")]
        name = target.strip().strip("`'\"()")
        if not name:
            return []
        short = re.split(r"[.:#]+", name)[-1]
        found: list[tuple[GraphNode, str]] = []
        for p, q in sorted(self.defs_by_name.get(short, ())):
            if q == name or q.endswith("." + name) or q.rsplit(".", 1)[-1] == short:
                node = self._def_node(p, q)
                if node is not None:
                    found.append((node, "exact_symbol"))
        if found:
            return found
        return [
            (self._file_node(p), "lexical") for p in sorted(self.referencers.get(short, ()))[:8]
        ]


_RELATION_WEIGHT = {
    "exact_symbol": 1.0,
    "exact_path": 1.0,
    "lexical": 0.45,
    "same_file": 0.8,
    "defines": 0.7,
    "contains": 0.72,
    "contained_by": 0.7,
    "defined_in": 0.75,
    "called_by": 0.7,
    "calls": 0.66,
    "imported_by": 0.65,
    "imports": 0.6,
    "includes": 0.6,
    "inherits": 0.6,
    "inherited_by": 0.6,
    "referenced_by": 0.4,
}


class CodeGraphAdapter:
    """Owner of per-worktree code graphs (see module docstring)."""

    def __init__(
        self,
        ledger: AgentLedgerDB | None = None,
        *,
        max_files: int = MAX_FILES,
        rescan_interval_s: float = RESCAN_INTERVAL_S,
        lister: Callable[[str], list[str] | None] | None = None,
    ) -> None:
        self.ledger = ledger
        self.max_files = max(1, int(max_files))
        self.rescan_interval_s = float(rescan_interval_s)
        self._lister = lister or _git_list_files
        self._graphs: OrderedDict[str, WorkspaceGraph] = OrderedDict()
        self._lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="headroom-code-graph"
        )
        self._pending: set[concurrent.futures.Future[Any]] = set()
        self._closed = False
        self.counters: dict[str, int] = {
            "full_scans": 0,
            "incremental_scans": 0,
            "files_parsed": 0,
            "files_reused": 0,
            "invalidations": 0,
            "errors": 0,
        }

    # -- lifecycle ----------------------------------------------------------

    def _submit(self, fn: Callable[..., Any], *args: Any) -> None:
        if self._closed:
            return
        try:
            future = self._executor.submit(self._guard, fn, *args)
        except RuntimeError:
            return
        with self._lock:
            self._pending.add(future)
        future.add_done_callback(lambda f: self._discard(f))

    def _discard(self, future: concurrent.futures.Future[Any]) -> None:
        with self._lock:
            self._pending.discard(future)

    def _guard(self, fn: Callable[..., Any], *args: Any) -> None:
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001 - indexing failures never surface to requests
            self.counters["errors"] += 1
            log.warning("code graph indexing failed: %s", describe_exception(exc))

    def wait_idle(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                pending = list(self._pending)
            if not pending:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            concurrent.futures.wait(pending, timeout=remaining)

    def close(self) -> None:
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)

    # -- workspace management ----------------------------------------------

    def graph(self, workspace_id: str) -> WorkspaceGraph | None:
        with self._lock:
            graph = self._graphs.get(workspace_id)
            if graph is not None:
                self._graphs.move_to_end(workspace_id)
            return graph

    def ensure_workspace(self, identity: WorkspaceIdentity) -> WorkspaceGraph | None:
        """Register a trusted worktree and schedule (re)indexing. Never blocks."""
        if not identity.can_authorize_paths() or not identity.root:
            return None
        now = time.monotonic()
        with self._lock:
            graph = self._graphs.get(identity.workspace_id)
            if graph is None:
                graph = WorkspaceGraph(identity.workspace_id, identity.root)
                self._graphs[identity.workspace_id] = graph
                while len(self._graphs) > MAX_WORKSPACES:
                    self._graphs.popitem(last=False)
                graph.indexing = True
                schedule = True
            else:
                self._graphs.move_to_end(identity.workspace_id)
                schedule = not graph.indexing and now - graph.last_scan >= self.rescan_interval_s
                if schedule:
                    graph.indexing = True
        if schedule:
            self._submit(self._scan, graph)
        return graph

    def invalidate_file(self, workspace_id: str, path: str) -> bool:
        """Re-parse one changed file in the background (no repo-wide scan)."""
        graph = self.graph(workspace_id)
        if graph is None:
            return False
        self.counters["invalidations"] += 1
        self._submit(self._reparse, graph, path)
        return True

    # -- indexer thread -----------------------------------------------------

    def _load_persisted(self, graph: WorkspaceGraph) -> dict[str, FileRecord]:
        if self.ledger is None:
            return {}
        rows = self.ledger.query(
            "SELECT path, hash, mtime_ns, size, lang, parser, payload FROM graph_files "
            "WHERE workspace_id=?",
            (graph.workspace_id,),
        )
        out: dict[str, FileRecord] = {}
        for row in rows:
            try:
                out[row[0]] = FileRecord.from_row(*row)
            except (ValueError, TypeError, KeyError):
                continue
        return out

    def _persist(
        self, graph: WorkspaceGraph, records: list[FileRecord], removed: list[str]
    ) -> None:
        if self.ledger is None or (not records and not removed):
            return
        now = time.time()
        statements: list[tuple[str, Sequence[Any]]] = [
            (
                "INSERT INTO graph_files(workspace_id, path, hash, mtime_ns, size, lang, parser, "
                "payload, indexed_at) VALUES (?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(workspace_id, path) DO UPDATE SET hash=excluded.hash, "
                "mtime_ns=excluded.mtime_ns, size=excluded.size, lang=excluded.lang, "
                "parser=excluded.parser, payload=excluded.payload, indexed_at=excluded.indexed_at",
                (
                    graph.workspace_id,
                    r.path,
                    r.hash,
                    r.mtime_ns,
                    r.size,
                    r.lang,
                    r.parser,
                    r.payload(),
                    now,
                ),
            )
            for r in records
        ]
        statements.extend(
            ("DELETE FROM graph_files WHERE workspace_id=? AND path=?", (graph.workspace_id, p))
            for p in removed
        )
        self.ledger.transaction(statements)

    def _candidate_paths(self, root: str) -> list[str]:
        listed = self._lister(root)
        if listed is None:
            listed = _walk_files(root, self.max_files * 2)
        exts = _source_extensions() | set(_ext_to_lang())
        ignore = _ignore_dirs()
        out: list[str] = []
        for rel in listed:
            rel = rel.replace("\\", "/")
            parts = rel.split("/")
            lowered = [p.lower() for p in parts[:-1]]
            if any(p in ignore or p in _GENERATED_SEGMENTS or p.startswith(".") for p in lowered):
                continue
            name = parts[-1].lower()
            if name.endswith(_GENERATED_SUFFIXES):
                continue
            if Path(name).suffix not in exts:
                continue
            out.append(rel)
            if len(out) >= self.max_files:
                break
        return sorted(out)

    def _scan(self, graph: WorkspaceGraph) -> None:
        try:
            first = not graph.ready
            persisted = self._load_persisted(graph) if first else {}
            paths = self._candidate_paths(graph.root)
            lang_map = _ext_to_lang()
            changed: list[FileRecord] = []
            seen = set(paths)
            for rel in paths:
                record = self._refresh(graph, rel, persisted.get(rel), lang_map)
                if record is not None:
                    changed.append(record)
            removed = [p for p in list(graph.files) if p not in seen]
            removed += [p for p in persisted if p not in seen and p not in graph.files]
            for path in removed:
                graph.remove(path)
            with graph.lock:
                for rel in paths:
                    graph._add_path(rel)
            self._persist(graph, changed, sorted(set(removed)))
            self.counters["full_scans" if first else "incremental_scans"] += 1
            graph.ready = True
        finally:
            graph.last_scan = time.monotonic()
            graph.indexing = False

    def _refresh(
        self,
        graph: WorkspaceGraph,
        rel: str,
        persisted: FileRecord | None,
        lang_map: dict[str, str],
    ) -> FileRecord | None:
        """Re-parse ``rel`` only when its stat or content hash changed."""
        full = os.path.join(graph.root, rel)
        try:
            st = os.stat(full)
        except OSError:
            return None
        if st.st_size > MAX_FILE_BYTES:
            return None
        current = graph.files.get(rel) or persisted
        if (
            current is not None
            and current.mtime_ns == st.st_mtime_ns
            and current.size == st.st_size
        ):
            if rel not in graph.files:
                graph.put(current)
                self.counters["files_reused"] += 1
            return None
        try:
            with open(full, "rb") as handle:
                data = handle.read(MAX_FILE_BYTES + 1)
        except OSError:
            return None
        if len(data) > MAX_FILE_BYTES or b"\x00" in data[:4096]:
            return None
        digest = hashlib.sha256(data).hexdigest()
        if current is not None and current.hash == digest:
            current.mtime_ns = st.st_mtime_ns
            current.size = st.st_size
            if rel not in graph.files:
                graph.put(current)
            self.counters["files_reused"] += 1
            return current
        lang = lang_map.get(Path(rel).suffix.lower())
        record = extract_file(rel, data, lang=lang)
        record.mtime_ns = st.st_mtime_ns
        record.size = st.st_size
        graph.put(record)
        graph.parse_counts[record.parser] = graph.parse_counts.get(record.parser, 0) + 1
        self.counters["files_parsed"] += 1
        return record

    def _reparse(self, graph: WorkspaceGraph, path: str) -> None:
        rel = graph.normalize_path(path)
        if rel is None:
            candidate = path[5:] if path.startswith("path:") else path
            candidate = candidate.replace("\\", "/")
            if os.path.isabs(candidate) or candidate.startswith(".."):
                return
            rel = candidate
        full = os.path.join(graph.root, rel)
        if not os.path.exists(full):
            graph.remove(rel)
            self._persist(graph, [], [rel])
            return
        exts = _source_extensions() | set(_ext_to_lang())
        if Path(rel).suffix.lower() not in exts:
            return
        record = self._refresh(graph, rel, None, _ext_to_lang())
        if record is not None:
            self._persist(graph, [record], [])

    # -- query API (request path) --------------------------------------------

    def neighbors(
        self,
        workspace_id: str,
        target: str,
        *,
        radius: int = DEFAULT_RADIUS,
        limit: int = 24,
        active_path: str | None = None,
    ) -> list[Neighbor]:
        """Ranked neighborhood of a symbol or path, scoped to one trusted workspace.

        Order: the exact symbol/path, the same file, then adjacent callers /
        imports / inheritance, then breadth ``radius``.
        """
        graph = self.graph(workspace_id)
        if graph is None or not graph.ready:
            return []
        radius = max(0, min(int(radius), 4))
        with graph.lock:
            starts = graph.resolve_targets(target)
            if not starts:
                return []
            active = graph.normalize_path(active_path) if active_path else None
            best: dict[str, Neighbor] = {}
            queue: deque[tuple[GraphNode, int, str, float]] = deque()
            for node, relation in starts:
                score = _RELATION_WEIGHT[relation] + (0.05 if node.path == active else 0.0)
                queue.append((node, 0, relation, score))
            visited = 0
            while queue and visited < MAX_VISITED:
                node, distance, relation, score = queue.popleft()
                current = best.get(node.id)
                if current is not None and current.score >= score:
                    continue
                best[node.id] = Neighbor(node, distance, relation, score)
                visited += 1
                if distance >= radius:
                    continue
                for edge, nxt in graph.adjacent(node):
                    edge_relation = edge
                    if (
                        distance == 0
                        and nxt.path is not None
                        and nxt.path == node.path
                        and nxt.kind != "file"
                    ):
                        edge_relation = "same_file"
                    weight = _RELATION_WEIGHT.get(edge_relation, 0.4)
                    queue.append((nxt, distance + 1, edge_relation, score * weight))
        ranked = sorted(best.values(), key=lambda n: (n.distance, -n.score, n.node.id))
        return ranked[: max(1, int(limit))]

    def symbols_for_file(self, workspace_id: str, path: str) -> list[GraphNode]:
        graph = self.graph(workspace_id)
        if graph is None:
            return []
        with graph.lock:
            rel = graph.normalize_path(path)
            record = graph.files.get(rel) if rel else None
            if record is None:
                return []
            return [
                GraphNode(f"sym:{rel}#{d.qualname}", d.kind, d.qualname, rel, d.line)
                for d in record.defs
            ]

    def affected_resources(
        self, workspace_id: str, paths: Iterable[str], *, radius: int = DEFAULT_RADIUS
    ) -> list[str]:
        """Resource ids of files that depend on ``paths`` (importers, callers, subclasses)."""
        graph = self.graph(workspace_id)
        if graph is None or not graph.ready:
            return []
        out: dict[str, int] = {}
        with graph.lock:
            frontier: list[str] = []
            for value in paths:
                rel = graph.normalize_path(value)
                if rel is not None:
                    frontier.append(rel)
            seen = set(frontier)
            for depth in range(1, max(1, radius) + 1):
                nxt: list[str] = []
                for rel in frontier:
                    dependents = set(graph.importers.get(rel, ()))
                    record = graph.files.get(rel)
                    if record is not None:
                        for d in record.defs:
                            short = d.qualname.rsplit(".", 1)[-1]
                            dependents |= {p for p, _ in graph.callers.get(short, ())}
                            dependents |= {p for p, _ in graph.subclasses.get(short, ())}
                    for dep in sorted(dependents):
                        if dep not in seen:
                            seen.add(dep)
                            out[dep] = depth
                            nxt.append(dep)
                frontier = nxt
                if not frontier:
                    break
        return [f"path:{p}" for p, _ in sorted(out.items(), key=lambda kv: (kv[1], kv[0]))]

    def status(self) -> dict[str, Any]:
        with self._lock:
            graphs = list(self._graphs.values())
        return {
            "workspaces": len(graphs),
            "files": sum(len(g.files) for g in graphs),
            "ready": sum(1 for g in graphs if g.ready),
            "counters": dict(self.counters),
        }


def _git_list_files(root: str) -> list[str] | None:
    """Tracked + untracked-not-ignored files (honours .gitignore); None without git."""
    if not os.path.exists(os.path.join(root, ".git")):
        return None
    from headroom._subprocess import run

    try:
        result = run(
            ["git", "-C", root, "ls-files", "-co", "--exclude-standard", "-z"],
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, ValueError) as exc:
        log.debug("git ls-files unavailable: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001 - subprocess.TimeoutExpired and friends
        log.debug("git ls-files failed: %s", exc)
        return None
    if result.returncode != 0:
        return None
    raw = result.stdout if isinstance(result.stdout, bytes) else str(result.stdout).encode()
    return [p.decode("utf-8", "replace") for p in raw.split(b"\x00") if p]


def _walk_files(root: str, cap: int) -> list[str]:
    ignore = _ignore_dirs()
    out: list[str] = []
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(
            d
            for d in dirs
            if d not in ignore and not d.startswith(".") and d.lower() not in _GENERATED_SEGMENTS
        )
        for name in sorted(files):
            rel = os.path.relpath(os.path.join(current, name), root).replace("\\", "/")
            out.append(rel)
            if len(out) >= cap:
                return out
    return out

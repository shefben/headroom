"""Graph/symbol-scoped code context (#11): tree-sitter reuse, ranking, incremental
updates, ignored paths, persistence, lexical fallback and workspace scoping."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from headroom.agent.code_graph import CodeGraphAdapter, extract_file
from headroom.agent.identity import WorkspaceIdentity
from headroom.agent.ledger_db import AgentLedgerDB

FILES = {
    "pkg/__init__.py": "",
    "pkg/parser.py": (
        "from pkg.lexer import tokenize\n\n"
        "class Parser(BaseParser):\n"
        "    def parse(self, text):\n"
        "        return decode_escape(tokenize(text))\n\n"
        "def decode_escape(tokens):\n"
        "    return [t for t in tokens]\n"
    ),
    "pkg/lexer.py": "def tokenize(text):\n    return text.split()\n",
    "pkg/base.py": "class BaseParser:\n    pass\n",
    "app/main.py": (
        "from pkg.parser import Parser\n\ndef run():\n    return Parser().parse('x')\n"
    ),
    "web/ui.ts": (
        "import { helper } from './util';\n"
        "export class Widget extends Base { render() { return helper(); } }\n"
    ),
    "web/util.ts": "export function helper() { return 1; }\n",
    "scripts/tool.lua": "function do_thing()\n  return decode_escape\nend\n",
    "node_modules/dep/index.js": "function hidden() {}\n",
    "vendor/lib.py": "def vendored():\n    pass\n",
    "build/gen.py": "def generated():\n    pass\n",
}


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _identity(root: Path, ws: str = "ws_graph") -> WorkspaceIdentity:
    return WorkspaceIdentity(ws, "ss_graph", True, "me", root=str(root))


@pytest.fixture()
def repo(tmp_path):
    root = tmp_path / "repo"
    _write(root, FILES)
    return root


def _adapter(ledger=None, **kw):
    # No git in the tmp repo: the adapter falls back to an ignore-aware walk.
    return CodeGraphAdapter(ledger, lister=lambda root: None, rescan_interval_s=0.0, **kw)


def _ready(adapter, identity):
    adapter.ensure_workspace(identity)
    assert adapter.wait_idle(30)
    graph = adapter.graph(identity.workspace_id)
    assert graph is not None and graph.ready
    return graph


def test_index_skips_ignored_generated_and_vendor_paths(repo):
    adapter = _adapter()
    graph = _ready(adapter, _identity(repo))
    assert "pkg/parser.py" in graph.files and "web/ui.ts" in graph.files
    assert not any(p.startswith(("node_modules/", "vendor/", "build/")) for p in graph.files)
    assert graph.files["pkg/parser.py"].parser == "tree-sitter"
    assert graph.files["scripts/tool.lua"].parser in ("lexical", "tree-sitter", "ast-grep")
    adapter.close()


def test_neighbors_rank_exact_then_same_file_then_callers(repo):
    adapter = _adapter()
    identity = _identity(repo)
    _ready(adapter, identity)
    result = adapter.neighbors(identity.workspace_id, "decode_escape", radius=2)
    assert result[0].relation == "exact_symbol" and result[0].node.name == "decode_escape"
    distances = [n.distance for n in result]
    assert distances == sorted(distances)
    by_name = {n.node.name: n for n in result}
    # The caller Parser.parse (same file) is adjacent; breadth 2 reaches importers.
    assert by_name["Parser.parse"].distance == 1
    assert any(n.node.path == "app/main.py" for n in result)
    # The lexical reference from an unsupported-language file is still found.
    assert any(n.node.path == "scripts/tool.lua" for n in result)
    # Radius bounds breadth.
    assert all(
        n.distance <= 1 for n in adapter.neighbors(identity.workspace_id, "decode_escape", radius=1)
    )
    adapter.close()


def test_path_targets_imports_inheritance_and_ts(repo):
    adapter = _adapter()
    identity = _identity(repo)
    _ready(adapter, identity)
    hood = adapter.neighbors(identity.workspace_id, str(repo / "pkg" / "parser.py"))
    assert hood[0].relation == "exact_path" and hood[0].node.path == "pkg/parser.py"
    relations = {(n.relation, n.node.path) for n in hood}
    assert ("imports", "pkg/lexer.py") in relations
    assert ("imported_by", "app/main.py") in relations
    parser_hood = adapter.neighbors(identity.workspace_id, "Parser")
    assert any(n.relation == "inherits" and n.node.name == "BaseParser" for n in parser_hood)
    ts = adapter.neighbors(identity.workspace_id, "web/ui.ts")
    assert any(n.node.path == "web/util.ts" and n.relation == "imports" for n in ts)
    symbols = {n.name for n in adapter.symbols_for_file(identity.workspace_id, "pkg/parser.py")}
    assert {"Parser", "Parser.parse", "decode_escape"} <= symbols
    affected = adapter.affected_resources(identity.workspace_id, ["pkg/lexer.py"])
    assert affected[0] == "path:pkg/parser.py" and "path:app/main.py" in affected
    adapter.close()


def test_incremental_invalidation_reparses_only_the_changed_file(repo):
    adapter = _adapter()
    identity = _identity(repo)
    graph = _ready(adapter, identity)
    parsed_before = adapter.counters["files_parsed"]
    hash_before = {p: r.hash for p, r in graph.files.items()}
    target = repo / "pkg" / "lexer.py"
    target.write_text(
        "def tokenize(text):\n    return lex(text)\n\ndef lex(t):\n    return t\n", encoding="utf-8"
    )
    assert adapter.invalidate_file(identity.workspace_id, "path:pkg/lexer.py")
    assert adapter.wait_idle(30)
    assert adapter.counters["files_parsed"] == parsed_before + 1
    assert graph.files["pkg/lexer.py"].hash != hash_before["pkg/lexer.py"]
    assert all(graph.files[p].hash == h for p, h in hash_before.items() if p != "pkg/lexer.py")
    assert "lex" in {
        n.name for n in adapter.symbols_for_file(identity.workspace_id, "pkg/lexer.py")
    }
    # A deleted file disappears from the graph.
    (repo / "web" / "util.ts").unlink()
    adapter.invalidate_file(identity.workspace_id, "web/util.ts")
    assert adapter.wait_idle(30)
    assert "web/util.ts" not in graph.files
    # A rescan with nothing changed re-parses nothing.
    parsed = adapter.counters["files_parsed"]
    adapter.ensure_workspace(identity)
    assert adapter.wait_idle(30)
    assert adapter.counters["files_parsed"] == parsed
    assert adapter.counters["incremental_scans"] >= 1
    adapter.close()


def test_persisted_index_survives_restart(repo, tmp_path):
    db = AgentLedgerDB(tmp_path / "agent_ledger.db")
    identity = _identity(repo)
    first = _adapter(db)
    _ready(first, identity)
    parsed = first.counters["files_parsed"]
    assert parsed >= 6
    first.close()
    second = _adapter(db)
    graph = _ready(second, identity)
    assert second.counters["files_parsed"] == 0
    assert second.counters["files_reused"] == parsed
    assert "decode_escape" in graph.defs_by_name
    # A file changed while the proxy was down is re-parsed alone.
    later = time.time() + 5
    lexer = repo / "pkg" / "lexer.py"
    lexer.write_text("def tokenize(text):\n    return []\n", encoding="utf-8")
    os.utime(lexer, (later, later))
    third = _adapter(db)
    _ready(third, identity)
    assert third.counters["files_parsed"] == 1
    second.close()
    third.close()
    db.close()


def test_untrusted_and_other_workspaces_are_isolated(repo, tmp_path):
    adapter = _adapter()
    untrusted = WorkspaceIdentity("wu_x", "su_x", False, "me")
    assert adapter.ensure_workspace(untrusted) is None
    other_root = tmp_path / "other"
    _write(other_root, {"only_here.py": "def lonely():\n    pass\n"})
    mine, theirs = _identity(repo), _identity(other_root, "ws_other")
    _ready(adapter, mine)
    _ready(adapter, theirs)
    assert adapter.neighbors(mine.workspace_id, "lonely") == []
    assert adapter.neighbors(theirs.workspace_id, "lonely")[0].node.path == "only_here.py"
    assert adapter.neighbors("ws_unknown", "decode_escape") == []
    adapter.close()


def test_extract_file_fallbacks():
    record = extract_file(
        "x.py", b"import os\nclass A(B):\n    def f(self):\n        g()\n", lang="python"
    )
    assert record.parser == "tree-sitter"
    assert {d.qualname for d in record.defs} == {"A", "A.f"}
    assert ("A.f", "g") in record.calls and ("A", "B") in record.bases
    assert record.imports == ["os"]
    lexical = extract_file("x.ex", b"defmodule Foo do\n  def bar, do: 1\nend\n", lang=None)
    assert lexical.parser == "lexical"
    assert "bar" in {d.qualname for d in lexical.defs}

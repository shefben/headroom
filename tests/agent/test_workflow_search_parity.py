"""Workflow Grep/Glob steps reproduce Claude Code's Grep/Glob (unified-21 #20).

The expected results below were observed by calling the real Claude Code
2.1.301 ``Grep`` and ``Glob`` tools (Windows 10, 2026-10-15) on the tree that
:func:`search_parity_fixture.build` creates. The workflow step dialect must
return the same matches for every semantic dimension: regex flavour, glob
semantics, include rules, hidden / gitignored / VCS files, case sensitivity,
binary files, ordering where the host defines one, pagination, path form,
long lines and symlinks. Patterns whose host behaviour Headroom cannot
reproduce are ineligible, and a missing ``rg`` makes Grep/Glob ineligible
rather than falling back to another matcher.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from headroom.agent import workflow_hosts as wh
from headroom.agent.workflow_macros import Safety, normalize_call

from .search_parity_fixture import build

pytestmark = pytest.mark.skipif(
    wh.search_unsupported_reason() is not None,
    reason="ripgrep not available: Grep/Glob steps are ineligible (tested separately)",
)

# -- observed host results (Claude Code 2.1.301) -------------------------------

#: Grep "needle", default files_with_matches: newest first; hidden files in,
#: gitignored / binary / .git / symlinked / case-mismatched files out.
HOST_GREP_FILES = [
    "src/c.txt",
    "UP.PY",
    "long.txt",
    "latin.txt",
    "node_modules/pkg/n.js",
    ".dotfile",
    ".hid/h.txt",
    "src/a.py",
]
#: Grep -i "needle" content: (file, line, text); the host's file order is
#: ripgrep's parallel walk order (unspecified), lines ascend within a file.
HOST_GREP_CONTENT_I = {
    ("UP.PY", 1, "needle upper"),
    ("src/sub/b.py", 1, "NEEDLE three"),
    ("src/c.txt", 1, "needle newest"),
    ("src/a.py", 1, "alpha Needle one"),
    ("src/a.py", 2, "needle two"),
    ("node_modules/pkg/n.js", 1, "needle nm"),
    ("long.txt", 1, "[Omitted long matching line]"),
    ("latin.txt", 1, "needle caf� latin1"),
    (".hid/h.txt", 1, "needle hidden"),
    (".dotfile", 1, "needle dot"),
}
HOST_GLOB_PY = ["src/a.py", "src/sub/b.py", "ignored_dir/i.py"]  # oldest first, no ignore
HOST_GLOB_TXT = [".hid/h.txt", "latin.txt", "long.txt", "la.txt", "src/c.txt"]


@pytest.fixture(scope="module")
def tree(tmp_path_factory) -> tuple[Path, dict[str, bool]]:
    base = tmp_path_factory.mktemp("parity")
    root = Path(os.path.realpath(base / "ws"))
    links = build(root)
    return root, links


def _run(root: Path, tool: str, args: dict) -> wh.StepRun:
    step = normalize_call(tool, args, str(root))
    assert step.template.safety == Safety.READ_ONLY.value, step.template.family
    dialect = wh.dialect(step.template.dialect)
    assert dialect is not None
    name, tool_input = dialect.compile_input(step.template, step.values, str(root))
    assert name == tool
    return dialect.run(step.template.kind, str(root), tool_input, 30.0)


def _files(run: wh.StepRun) -> list[str]:
    assert run.ok, run.error
    lines = run.text.splitlines()
    if lines and re.match(r"^Found \d+ files?$", lines[0]):
        lines = lines[1:]
    if lines == ["No files found"]:
        return []
    return [ln for ln in lines if ln and not ln.startswith(("[", "("))]


def _content(run: wh.StepRun) -> list[tuple[str, int, str]]:
    assert run.ok, run.error
    out = []
    for line in run.text.splitlines():
        m = re.match(r"^(.+?):(\d+):(.*)$", line)
        if m:
            out.append((m.group(1), int(m.group(2)), m.group(3)))
    return out


def test_grep_files_with_matches_matches_host(tree):
    root, _ = tree
    run = _run(root, "Grep", {"pattern": "needle", "path": str(root)})
    assert _files(run) == HOST_GREP_FILES  # same set, same newest-first order
    assert run.text.startswith("Found 8 files")


def test_grep_content_case_insensitive_matches_host(tree):
    root, _ = tree
    run = _run(
        root, "Grep", {"pattern": "needle", "path": str(root), "-i": True, "output_mode": "content"}
    )
    rows = _content(run)
    assert set(rows) == HOST_GREP_CONTENT_I
    assert len(rows) == len(HOST_GREP_CONTENT_I)
    # Lines ascend within each file (the host's only ordering guarantee here);
    # Headroom's file order is deterministic (path order).
    assert rows == sorted(rows, key=lambda r: (r[0], r[1]))


def test_grep_case_sensitivity_glob_type_and_count_match_host(tree):
    root, _ = tree
    sensitive = _files(_run(root, "Grep", {"pattern": "NEEDLE", "path": str(root)}))
    assert sensitive == ["src/sub/b.py"]
    upper = _files(_run(root, "Grep", {"pattern": "needle", "path": str(root), "glob": "*.PY"}))
    assert upper == ["UP.PY"]  # glob is case-sensitive on every platform
    typed = _files(_run(root, "Grep", {"pattern": "needle", "path": str(root), "type": "py"}))
    assert typed == ["src/a.py"]
    count = _run(
        root,
        "Grep",
        {"pattern": "needle", "path": str(root), "output_mode": "count", "glob": "*.py"},
    )
    assert count.ok and count.text.splitlines()[0] == "src/a.py:1"
    assert "Found 1 total occurrence across 1 file." in count.text


def test_grep_line_numbers_path_scope_multiline_and_pagination_match_host(tree):
    root, _ = tree
    plain = _run(
        root,
        "Grep",
        {"pattern": "needle", "path": str(root / "src"), "output_mode": "content", "-n": False},
    )
    assert plain.ok
    assert set(plain.text.splitlines()) == {"src/c.txt:needle newest", "src/a.py:needle two"}
    multi = _run(
        root,
        "Grep",
        {
            "pattern": "foobar\\nfoobaz",
            "path": str(root),
            "multiline": True,
            "output_mode": "content",
        },
    )
    assert _content(multi) == [("la.txt", 1, "foobar"), ("la.txt", 2, "foobaz")]
    page = _run(
        root, "Grep", {"pattern": "needle", "path": str(root), "head_limit": 2, "offset": 1}
    )
    assert _files(page) == ["UP.PY", "long.txt"] and page.partial
    default = _run(root, "Grep", {"pattern": "needle", "path": str(root), "head_limit": 0})
    assert _files(default) == HOST_GREP_FILES and not default.partial


def test_grep_regex_flavour_is_ripgrep(tree):
    root, _ = tree
    # Look-around is a Python-re feature ripgrep rejects; the host errors too.
    look = _run(root, "Grep", {"pattern": "foo(?=bar)", "path": str(root)})
    assert not look.ok and "look-around" in (look.error or "")
    # A Unicode class Python's re does not know (it would reject the pattern).
    rust = _files(_run(root, "Grep", {"pattern": r"\p{Lu}EEDLE", "path": str(root)}))
    assert rust == ["src/sub/b.py"]
    word = _files(_run(root, "Grep", {"pattern": r"\bNeedle\b", "path": str(root)}))
    assert word == ["src/a.py"]


def test_binary_explicit_file_matches_host(tree):
    root, _ = tree
    run = _run(
        root,
        "Grep",
        {"pattern": "needle", "path": str(root / "bin.dat"), "output_mode": "content"},
    )
    assert run.ok
    assert run.text.strip() == 'binary file matches (found "\\0" byte around offset 6)'


def test_glob_matches_host(tree):
    root, _ = tree
    assert _files(_run(root, "Glob", {"pattern": "*.py", "path": str(root)})) == HOST_GLOB_PY
    assert _files(_run(root, "Glob", {"pattern": "**/*.txt", "path": str(root)})) == HOST_GLOB_TXT
    sub = _files(_run(root, "Glob", {"pattern": "**/sub/*.py", "path": str(root)}))
    assert sub == ["src/sub/b.py"]
    upper = _files(_run(root, "Glob", {"pattern": "*.PY", "path": str(root)}))
    assert upper == ["UP.PY"]


def test_symlinks_are_not_followed(tree):
    root, links = tree
    if not any(links.values()):
        pytest.skip("symlinks cannot be created on this platform/account")
    assert _files(_run(root, "Grep", {"pattern": "needle outside", "path": str(root)})) == []
    names = _files(_run(root, "Glob", {"pattern": "**/*.txt", "path": str(root)}))
    assert "linkfile.txt" not in names and not any(n.startswith("linkdir/") for n in names)


def test_glob_truncates_at_host_limit(tmp_path):
    root = Path(os.path.realpath(tmp_path))
    for i in range(wh.GLOB_MAX_RESULTS + 5):
        path = root / f"f{i:03}.md"
        path.write_text("x")
        os.utime(path, (1_700_000_000 + i, 1_700_000_000 + i))
    run = _run(root, "Glob", {"pattern": "*.md", "path": str(root)})
    files = _files(run)
    assert len(files) == wh.GLOB_MAX_RESULTS and files[0] == "f000.md" and run.partial
    assert f"(Showing {wh.GLOB_MAX_RESULTS} of {wh.GLOB_MAX_RESULTS + 5}" in run.text


def test_unreproducible_patterns_are_ineligible(tree):
    root, _ = tree
    cases = [
        ("Glob", {"pattern": "src/*.py", "path": str(root)}),  # host roots slash globs differently
        ("Grep", {"pattern": "needle", "path": str(root), "glob": "src/*.py"}),
        ("Grep", {"pattern": "needle", "path": str(root), "-A": 2}),  # context: unverified
        ("Grep", {"pattern": "-needle", "path": str(root)}),
        ("Grep", {"pattern": "needle", "path": str(root.parent)}),  # outside the workspace
        ("grep", {"pattern": "needle", "include": "*.py"}),  # OpenCode: no dialect
        ("glob", {"pattern": "*.py"}),
    ]
    for tool, args in cases:
        step = normalize_call(tool, args, str(root))
        assert step.template.safety == Safety.UNSUPPORTED.value, (tool, args)


def test_missing_ripgrep_makes_grep_and_glob_ineligible(tree, monkeypatch):
    root, _ = tree
    import headroom.agent.exec_runner as er

    real = er.resolve_executable
    monkeypatch.setattr(er, "resolve_executable", lambda name: None if name == "rg" else real(name))
    for tool in ("Grep", "Glob"):
        step = normalize_call(tool, {"pattern": "*.py" if tool == "Glob" else "x"}, str(root))
        assert step.template.safety == Safety.UNSUPPORTED.value
        assert step.template.family.endswith("ripgrep_unavailable")
    template = normalize_call("Read", {"file_path": str(root / "src/a.py")}, str(root)).template
    assert wh.dialect(wh.CLAUDE_DIALECT).fingerprint(template) is not None  # Read needs no rg
    run = wh.dialect(wh.CLAUDE_DIALECT).run("grep", str(root), {"pattern": "x"}, 5.0)
    assert not run.ok and "ripgrep_unavailable" in (run.error or "")


def test_ripgrep_config_makes_steps_ineligible(tree, monkeypatch):
    root, _ = tree
    monkeypatch.setenv("RIPGREP_CONFIG_PATH", str(root / ".gitignore"))
    step = normalize_call("Grep", {"pattern": "x"}, str(root))
    assert step.template.family.endswith("ripgrep_config_present")


def test_host_version_gate():
    dialect = wh.dialect(wh.CLAUDE_DIALECT)
    assert dialect is not None
    grep = normalize_call("Grep", {"pattern": "x"}, None).template
    assert wh.host_version_of({"agent": "claude-code_2-1-300_agent"}) == "2.1.300"
    assert dialect.host_semantics_ok(grep, "2.1.300")
    assert not dialect.host_semantics_ok(grep, "2.2.0")
    assert not dialect.host_semantics_ok(grep, None)

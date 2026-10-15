"""The seven tool-family interceptors behind one dispatcher (unified-21 #14).

Each interceptor turns one :class:`ToolResultEnvelope` into a *structural
view* candidate. It never stores, gates or swaps anything itself: the
:class:`~headroom.agent.tool_admission.dispatcher.ToolAdmissionDispatcher`
indexes the exact original in CCR, appends the retrieval footer, and submits
the view through the SafetyGate ladder. Interceptors reuse the existing
components instead of re-implementing them:

========  ==================================================================
Read      code-graph extraction (tree-sitter -> ast-grep interceptor runner
          -> lexical) for an outline that keeps the client's own line
          numbers; edit targets, range reads and small reads stay exact.
Search    Rust ``SearchCompressor`` (grouped by file, task-ranked).
Listing   directory/package grouping (no existing compressor covers it).
Shell     Rust ``LogCompressor`` (errors, traces, first/last boundaries).
Test      Rust ``LogCompressor`` in its test-runner formats.
Git       Rust ``DiffCompressor`` for diffs; status grouping for status.
JSON      ``SmartCrusher`` keyed rows (lossless table when it round-trips).
========  ==================================================================

After every compressor, :func:`pin_repair` re-adds, verbatim and in original
order, any line the family must keep that the compressor dropped: user-named
entities, failure/exception lines, exit/status lines, test summaries,
failing test ids and diff headers. The SafetyGate remains the final judge.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Protocol

from headroom.agent.interfaces import ToolResultEnvelope
from headroom.agent.tool_admission.families import ToolFamily, path_of, workspace_relative

log = logging.getLogger(__name__)

#: Results below this size are left exact (no structural view is worth a footer).
MIN_VIEW_CHARS = 1500
MAX_PINNED_LINES = 80

_FAILURE_LINE = re.compile(
    r"(?:\b(?:error|errors|exception|traceback|fatal|panic(?:ked)?|failed|failure|"
    r"segmentation fault|assert(?:ion)?(?:error)?|denied|refused|not found|cannot|unable to)\b|"
    r"^\s*E\s{2,}|^FAILED\b|^ERROR\b|\bFAIL\b)",
    re.IGNORECASE,
)
_EXIT_LINE = re.compile(
    r"(?i)\b(?:exit(?:ed)?(?: with)? (?:code|status)|returned non-zero exit status|"
    r"process finished with exit code|exit_code)\b"
)
_SUMMARY_LINE = re.compile(
    r"\b\d+ (?:passed|failed|skipped|errors?|xfailed|xpassed|deselected|warnings?)\b|"
    r"\btest result: \w+\.|\bTests?:\s+\d+|\bTest Suites?:|^=+ .* in [\d.]+s",
    re.IGNORECASE,
)
_TEST_OUTCOME = re.compile(
    r"^(?P<id>\S+::\S+|\S+\.py::\S+)\s+(?P<outcome>PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b|"
    r"^(?P<outcome2>FAILED|ERROR|XPASS|XFAIL|SKIPPED)\s+(?P<id2>\S+::\S+)|"
    r"^test (?P<id3>\S+) \.\.\. (?P<outcome3>ok|FAILED|ignored)\b|"
    r"^\s*(?P<outcome4>✓|✕|×|PASS|FAIL)\s+(?P<id4>.+?)(?:\s+\(\d+\s*ms\))?$"
)
_OUTCOME_ALIASES = {
    "ok": "PASSED",
    "✓": "PASSED",
    "PASS": "PASSED",
    "✕": "FAILED",
    "×": "FAILED",
    "FAIL": "FAILED",
    "ignored": "SKIPPED",
}
_DIFF_HEADER = re.compile(r"^(?:diff --git |--- |\+\+\+ |@@ )")
_GUTTER = re.compile(r"^(\s*\d+)(\t|→|\|)")


@dataclass(frozen=True)
class ViewContext:
    """Request facts an interceptor may condition on (no I/O, no state)."""

    query: str = ""
    entities: tuple[str, ...] = ()
    working_set: frozenset[str] = frozenset()
    root: str | None = None
    cwd: str | None = None
    tool_input: Any = field(default_factory=dict)
    applyable_patch: bool = False


@dataclass(frozen=True)
class FamilyView:
    """A structural view of one result (body only; the dispatcher adds the footer)."""

    text: str
    label: str
    conservative: str | None = None
    reversible: bool = False
    content_type: str = "plain_text"


class ToolInterceptor(Protocol):
    family: ToolFamily

    def view(self, envelope: ToolResultEnvelope, ctx: ViewContext) -> FamilyView | None: ...


def _norm(line: str) -> str:
    return " ".join(line.split())


def must_keep_line(line: str, entities: Sequence[str]) -> bool:
    if any(e and e in line for e in entities):
        return True
    return bool(
        _FAILURE_LINE.search(line)
        or _EXIT_LINE.search(line)
        or _SUMMARY_LINE.search(line)
        or _TEST_OUTCOME.match(line.strip()) and re.search(r"FAIL|ERROR|XPASS|✕|×", line)
    )


def pin_repair(
    original: str,
    view: str,
    entities: Sequence[str],
    *,
    extra: re.Pattern[str] | None = None,
    limit: int = MAX_PINNED_LINES,
    facts: bool = True,
) -> str:
    """Re-add (verbatim, original order) must-keep lines the view dropped.

    ``facts=False`` pins only user-named entities (source code, listings),
    where the failure-word heuristics would just be noise.
    """
    present = {_norm(line) for line in view.splitlines()}
    missing: list[str] = []
    seen: set[str] = set()
    for line in original.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        key = _norm(line)
        if key in present or key in seen:
            continue
        keep = (
            must_keep_line(line, entities)
            if facts
            else any(e and e in line for e in entities)
        )
        if keep or (extra is not None and extra.search(line)):
            seen.add(key)
            missing.append(line)
            if len(missing) >= limit:
                break
    if not missing:
        return view
    sep = "" if view.endswith("\n") or not view else "\n"
    return f"{view}{sep}[headroom: lines kept verbatim from the original]\n" + "\n".join(missing)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def _strip_gutter(text: str) -> tuple[list[str], list[str], bool]:
    """``(display_lines, source_lines, numbered)`` for a client Read result.

    Claude Code (and others) number lines (``  12\\tcode``). Parsing runs on
    the source lines; the outline keeps the display lines so line numbers in
    the view are the client's own, never re-counted.
    """
    display = text.split("\n")
    body = [ln for ln in display if ln.strip()]
    hits = sum(1 for ln in body[:200] if _GUTTER.match(ln))
    numbered = bool(body) and hits >= max(3, int(0.8 * min(len(body), 200)))
    if not numbered:
        return display, display, False
    source = []
    for ln in display:
        m = _GUTTER.match(ln)
        source.append(ln[m.end() :] if m else ln)
    return display, source, True


class ReadInterceptor:
    family = ToolFamily.READ
    min_chars = 6000

    def view(self, envelope: ToolResultEnvelope, ctx: ViewContext) -> FamilyView | None:
        text = envelope.content
        if len(text) < self.min_chars:
            return None
        if envelope.partial_reason == "range_read":
            return None  # the model asked for exactly these lines
        args = ctx.tool_input if isinstance(ctx.tool_input, dict) else {}
        raw_path = path_of(args) or ""
        if not raw_path:
            return None
        rel = workspace_relative(raw_path, root=ctx.root, cwd=ctx.cwd)
        if rel in ctx.working_set or raw_path in ctx.working_set:
            return None  # current edit target stays exact
        from headroom.agent.code_graph import _ext_to_lang, extract_file

        lang = _ext_to_lang().get(PurePosixPath(raw_path.replace("\\", "/")).suffix.lower())
        if lang is None:
            return None  # non-code reads go to the content router
        display, source, _numbered = _strip_gutter(text)
        try:
            record = extract_file(raw_path, "\n".join(source).encode("utf-8"), lang=lang)
        except Exception as exc:  # noqa: BLE001 - no outline, original stays
            log.debug("read outline extraction failed: %s", exc)
            return None
        def_lines = sorted({d.line for d in record.defs if 1 <= d.line <= len(source)})
        named = self._named_spans(record.defs, def_lines, len(source), ctx.entities)
        if len(def_lines) < 3:
            return None
        aggressive = self._outline(display, source, def_lines, ctx.entities, named, body_lines=0)
        conservative = self._outline(display, source, def_lines, ctx.entities, named, body_lines=3)
        header = f"[headroom: outline of {rel} ({len(def_lines)} definitions"
        if envelope.is_partial:
            header += f"; PARTIAL read ({envelope.partial_reason}): visible portion only"
        header += "; bodies elided, line numbers are the file's own]"
        return FamilyView(
            text=f"{header}\n{aggressive}",
            conservative=f"{header}\n{conservative}",
            label="read_outline",
            content_type="source_code",
        )

    @staticmethod
    def _named_spans(
        defs: Sequence[Any], def_lines: list[int], total: int, entities: Sequence[str]
    ) -> set[int]:
        """0-based line indexes of the full bodies of user-named definitions.

        The exact active symbol is never hidden by an outline: a definition
        whose name the user wrote is kept whole (bounded per definition).
        """
        wanted = {e.rsplit(".", 1)[-1].rstrip("()") for e in entities if e}
        keep: set[int] = set()
        for d in defs:
            name = str(getattr(d, "qualname", "")).rsplit(".", 1)[-1]
            if not name or name not in wanted:
                continue
            start = int(d.line)
            later = [ln for ln in def_lines if ln > start]
            end = min(later[0] - 1 if later else total, start + 199)
            keep.update(range(start - 1, end))
        return keep

    @staticmethod
    def _outline(
        display: list[str],
        source: list[str],
        def_lines: list[int],
        entities: Sequence[str],
        named: set[int],
        *,
        body_lines: int,
    ) -> str:
        keep: set[int] = {i for i in named if i < len(source)}
        for ln in def_lines:
            idx = ln - 1
            keep.add(idx)
            # decorators directly above, docstring/first lines below
            j = idx - 1
            while j >= 0 and source[j].lstrip().startswith("@"):
                keep.add(j)
                j -= 1
            for k in range(idx + 1, min(len(source), idx + 1 + body_lines)):
                keep.add(k)
            nxt = idx + 1
            while nxt < len(source) and not source[nxt].strip():
                nxt += 1
            if nxt < len(source) and source[nxt].lstrip().startswith(('"""', "'''", "/**", "///")):
                keep.add(nxt)
        first_def = def_lines[0] - 1
        for i, line in enumerate(source):
            stripped = line.strip()
            if not stripped:
                continue
            if i < first_def and i < 40:
                keep.add(i)  # module header: imports, constants
            elif not line.startswith((" ", "\t")) and re.match(r"^[A-Z][A-Z0-9_]{2,}\s*[:=]", stripped):
                keep.add(i)  # top-level constants
            elif any(e and e in line for e in entities):
                keep.add(i)
        out: list[str] = []
        last = -1
        for i in sorted(keep):
            if i - last > 1:
                gap = i - last - 1
                out.append(f"[... {gap} lines elided ...]")
            out.append(display[i])
            last = i
        if last < len(display) - 1:
            tail = [d for d in display[last + 1 :] if d.strip()]
            if tail:
                out.append(f"[... {len(display) - 1 - last} lines elided ...]")
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class SearchInterceptor:
    family = ToolFamily.SEARCH

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._compressors: dict[int, Any] = {}

    def _compressor(self, per_file: int) -> Any:
        with self._lock:
            comp = self._compressors.get(per_file)
            if comp is None:
                from headroom.transforms.search_compressor import (
                    SearchCompressor,
                    SearchCompressorConfig,
                )

                comp = SearchCompressor(
                    SearchCompressorConfig(
                        max_matches_per_file=per_file,
                        max_total_matches=per_file * 8,
                        max_files=20,
                        enable_ccr=False,
                        group_by_file=True,
                    )
                )
                self._compressors[per_file] = comp
            return comp

    def view(self, envelope: ToolResultEnvelope, ctx: ViewContext) -> FamilyView | None:
        text = envelope.content
        if len(text) < MIN_VIEW_CHARS:
            return None
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if lines and all(":" not in ln for ln in lines[:50]):
            return _listing_view(envelope, ctx, label="search_files_grouped")
        query = " ".join([*ctx.entities, ctx.query])[:1500]
        try:
            aggressive = self._compressor(5).compress(text, query).compressed
            conservative = self._compressor(12).compress(text, query).compressed
        except Exception as exc:  # noqa: BLE001
            log.debug("search view failed: %s", exc)
            return None
        files = len({ln.split(":", 1)[0] for ln in lines if ":" in ln})
        header = f"[headroom: {len(lines)} search lines in {files} files, grouped by file"
        if envelope.is_partial:
            header += f"; PARTIAL result ({envelope.partial_reason}): totals unknown"
        header += "; queried matches and failures kept]"
        return FamilyView(
            text=header + "\n" + pin_repair(text, aggressive, ctx.entities),
            conservative=header + "\n" + pin_repair(text, conservative, ctx.entities),
            label="search_grouped",
            content_type="search_results",
        )


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


def _group_paths(paths: list[str], *, per_dir: int) -> list[str]:
    groups: OrderedDict[str, list[str]] = OrderedDict()
    for p in paths:
        norm = p.replace("\\", "/")
        parent, _, name = norm.rstrip("/").rpartition("/")
        groups.setdefault(parent or ".", []).append(name + ("/" if norm.endswith("/") else ""))
    out: list[str] = []
    for parent, names in groups.items():
        shown = names[:per_dir]
        more = len(names) - len(shown)
        suffix = f" [+{more} more]" if more > 0 else ""
        out.append(f"{parent}/ ({len(names)}): {', '.join(shown)}{suffix}")
    return out


def _listing_view(envelope: ToolResultEnvelope, ctx: ViewContext, *, label: str) -> FamilyView | None:
    text = envelope.content
    if len(text) < MIN_VIEW_CHARS:
        return None
    entries = [ln.strip() for ln in text.splitlines() if ln.strip()]
    paths = [e for e in entries if not e.startswith("[") and len(e) < 1024]
    if len(paths) < 20:
        return None
    pinned = [p for p in paths if any(e and e in p for e in ctx.entities)]
    header = f"[headroom: {len(paths)} entries grouped by directory"
    if envelope.is_partial:
        header += f"; PARTIAL listing ({envelope.partial_reason}): not all files are shown"
    header += "]"
    body_a = _group_paths(paths, per_dir=8)
    body_c = _group_paths(paths, per_dir=30)
    pin = ("\n[headroom: named entries]\n" + "\n".join(pinned[:MAX_PINNED_LINES])) if pinned else ""
    return FamilyView(
        text=header + "\n" + "\n".join(body_a) + pin,
        conservative=header + "\n" + "\n".join(body_c) + pin,
        label=label,
        content_type="plain_text",
    )


class ListingInterceptor:
    family = ToolFamily.LISTING

    def view(self, envelope: ToolResultEnvelope, ctx: ViewContext) -> FamilyView | None:
        return _listing_view(envelope, ctx, label="listing_grouped")


# ---------------------------------------------------------------------------
# Shell / build / log and test
# ---------------------------------------------------------------------------


class _LogBacked:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._compressors: dict[int, Any] = {}

    def _compressor(self, total: int) -> Any:
        with self._lock:
            comp = self._compressors.get(total)
            if comp is None:
                from headroom.transforms.log_compressor import LogCompressor, LogCompressorConfig

                comp = LogCompressor(
                    LogCompressorConfig(
                        max_total_lines=total,
                        max_errors=max(10, total // 5),
                        enable_ccr=False,
                    )
                )
                self._compressors[total] = comp
            return comp

    def _views(self, text: str, ctx: ViewContext, extra: re.Pattern[str] | None) -> tuple[str, str]:
        aggressive = self._compressor(80).compress(text, ctx.query).compressed
        conservative = self._compressor(200).compress(text, ctx.query).compressed
        return (
            pin_repair(text, aggressive, ctx.entities, extra=extra),
            pin_repair(text, conservative, ctx.entities, extra=extra),
        )


class ShellInterceptor(_LogBacked):
    family = ToolFamily.SHELL

    def view(self, envelope: ToolResultEnvelope, ctx: ViewContext) -> FamilyView | None:
        text = envelope.content
        if len(text) < MIN_VIEW_CHARS or text.count("\n") < 30:
            return None
        try:
            aggressive, conservative = self._views(text, ctx, None)
        except Exception as exc:  # noqa: BLE001
            log.debug("shell view failed: %s", exc)
            return None
        header = f"[headroom: {envelope.line_count} output lines; progress folded"
        if envelope.exit_code is not None:
            header += f"; exit code {envelope.exit_code}"
        if envelope.is_partial:
            header += f"; PARTIAL output ({envelope.partial_reason})"
        header += "; errors, traces and boundaries kept]"
        return FamilyView(
            text=header + "\n" + aggressive,
            conservative=header + "\n" + conservative,
            label="shell_log",
            content_type="build_output",
        )


def parse_test_outcomes(text: str) -> dict[str, str]:
    """``test id -> outcome`` parsed from pytest/cargo/jest-style output."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = _TEST_OUTCOME.match(line.strip())
        if not m:
            continue
        tid = m.group("id") or m.group("id2") or m.group("id3") or m.group("id4")
        outcome = m.group("outcome") or m.group("outcome2") or m.group("outcome3") or m.group(
            "outcome4"
        )
        if not tid or not outcome:
            continue
        outcome = _OUTCOME_ALIASES.get(outcome, outcome)
        # A later FAILED line (short summary) never downgrades to PASSED.
        if out.get(tid) in ("FAILED", "ERROR") and outcome not in ("FAILED", "ERROR"):
            continue
        out[tid] = outcome
    return out


class TestRunInterceptor(_LogBacked):
    family = ToolFamily.TEST
    _ASSERT = re.compile(r"^\s*(?:E\s+|>\s+|assert\b|AssertionError|thread '.*' panicked|---- )")

    def view(self, envelope: ToolResultEnvelope, ctx: ViewContext) -> FamilyView | None:
        text = envelope.content
        if len(text) < MIN_VIEW_CHARS or text.count("\n") < 30:
            return None
        try:
            aggressive, conservative = self._views(text, ctx, self._ASSERT)
        except Exception as exc:  # noqa: BLE001
            log.debug("test view failed: %s", exc)
            return None
        outcomes = parse_test_outcomes(text)
        failing = sorted(t for t, o in outcomes.items() if o in ("FAILED", "ERROR"))
        header = "[headroom: test run"
        if envelope.exit_code is not None:
            header += f", exit code {envelope.exit_code}"
        if outcomes:
            header += f"; {len(outcomes)} test outcomes parsed, {len(failing)} failing"
        if envelope.is_partial:
            header += f"; PARTIAL output ({envelope.partial_reason}): counts may be incomplete"
        header += "; failures, assertions, tracebacks and summary kept]"
        extra = ""
        missing_ids = [t for t in failing if t not in aggressive]
        if missing_ids:
            extra = "\n[headroom: failing test ids]\n" + "\n".join(missing_ids[:MAX_PINNED_LINES])
        return FamilyView(
            text=header + "\n" + aggressive + extra,
            conservative=header + "\n" + conservative + extra,
            label="test_log",
            content_type="build_output",
        )


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------

_STATUS_LINE = re.compile(r"^(?:[ MADRCU?!]{2} |\t(?:modified|new file|deleted|renamed|both modified):)")


class GitInterceptor:
    family = ToolFamily.GIT

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._compressors: dict[int, Any] = {}

    def _compressor(self, context_lines: int) -> Any:
        with self._lock:
            comp = self._compressors.get(context_lines)
            if comp is None:
                from headroom.transforms.diff_compressor import (
                    DiffCompressor,
                    DiffCompressorConfig,
                )

                comp = DiffCompressor(
                    DiffCompressorConfig(
                        max_context_lines=context_lines,
                        max_hunks_per_file=10 if context_lines < 3 else 40,
                        max_files=20 if context_lines < 3 else 60,
                        enable_ccr=False,
                    )
                )
                self._compressors[context_lines] = comp
            return comp

    def view(self, envelope: ToolResultEnvelope, ctx: ViewContext) -> FamilyView | None:
        text = envelope.content
        if len(text) < MIN_VIEW_CHARS:
            return None
        if "diff --git " in text or text.lstrip().startswith(("--- ", "@@ ")):
            if ctx.applyable_patch:
                return None  # an applyable patch is needed: the patch stays unmodified
            try:
                aggressive = self._compressor(1).compress(text, ctx.query).compressed
                conservative = self._compressor(3).compress(text, ctx.query).compressed
            except Exception as exc:  # noqa: BLE001
                log.debug("git diff view failed: %s", exc)
                return None
            header = "[headroom: diff with selected hunks; file names, headers and changed lines kept"
            if envelope.is_partial:
                header += f"; PARTIAL diff ({envelope.partial_reason})"
            header += "; not an applyable patch]"
            return FamilyView(
                text=header + "\n" + pin_repair(text, aggressive, ctx.entities, extra=_DIFF_HEADER),
                conservative=header
                + "\n"
                + pin_repair(text, conservative, ctx.entities, extra=_DIFF_HEADER),
                label="git_diff_hunks",
                content_type="git_diff",
            )
        lines = text.splitlines()
        status = [ln for ln in lines if _STATUS_LINE.match(ln)]
        if len(status) < 20:
            return None
        groups: OrderedDict[str, list[str]] = OrderedDict()
        for ln in status:
            if ln.startswith("\t"):
                code, _, name = ln.strip().partition(":")
                name = name.strip()
            else:
                code, name = ln[:2], ln[3:]
            groups.setdefault(code.strip() or "?", []).append(name)
        body = [f"{code} ({len(names)}): {', '.join(names)}" for code, names in groups.items()]
        others = [ln for ln in lines if ln.strip() and not _STATUS_LINE.match(ln)]
        head = others[:12]
        header = f"[headroom: git status, {len(status)} paths grouped by status code"
        if envelope.is_partial:
            header += f"; PARTIAL ({envelope.partial_reason})"
        header += "; every path kept]"
        view = "\n".join([header, *head, *body])
        return FamilyView(text=view, conservative=None, label="git_status_grouped")


# ---------------------------------------------------------------------------
# JSON / API / DB
# ---------------------------------------------------------------------------


class JsonInterceptor:
    family = ToolFamily.JSON

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._crusher: Any = None

    def _get(self) -> Any:
        with self._lock:
            if self._crusher is None:
                from headroom.config import CCRConfig
                from headroom.transforms.smart_crusher import SmartCrusher

                self._crusher = SmartCrusher(ccr_config=CCRConfig(enabled=False))
            return self._crusher

    def view(self, envelope: ToolResultEnvelope, ctx: ViewContext) -> FamilyView | None:
        text = envelope.content
        if len(text) < MIN_VIEW_CHARS:
            return None
        query = " ".join([*ctx.entities, ctx.query])[:1500]
        try:
            crushed, modified, _info = self._get()._smart_crush_content(
                text, query_context=query, tool_name=envelope.tool_name or None
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("json view failed: %s", exc)
            return None
        if not modified or crushed == text:
            return None
        from headroom.transforms.content_router import _compaction_roundtrips

        reversible = _compaction_roundtrips(text, crushed)
        return FamilyView(
            text=crushed,
            conservative=None,
            label="json_keyed_rows",
            reversible=reversible,
            content_type="json_array",
        )


#: The one interceptor per family the dispatcher consults.
INTERCEPTORS: dict[ToolFamily, ToolInterceptor] = {
    ToolFamily.READ: ReadInterceptor(),
    ToolFamily.SEARCH: SearchInterceptor(),
    ToolFamily.LISTING: ListingInterceptor(),
    ToolFamily.SHELL: ShellInterceptor(),
    ToolFamily.TEST: TestRunInterceptor(),
    ToolFamily.GIT: GitInterceptor(),
    ToolFamily.JSON: JsonInterceptor(),
}

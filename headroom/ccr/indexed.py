"""Indexed / selective CCR retrieval (unified-21 #3).

Extends the existing CCR ``CompressionStore`` and ``headroom_retrieve`` tool
without replacing hashes or markers: ``headroom_retrieve(hash)`` still returns
the stored original byte-for-byte. Optional arguments add selective modes:

``mode="full"``      exact original (the legacy behavior; the default).
``mode="search"``    the ``top_k`` spans most relevant to ``query`` (default 5,
                     at most ``max_tokens`` = 1200 tokens), each with the exact
                     source hash, span id, ordinal, byte range, score, type and
                     key/path/symbol metadata, plus ``has_more`` / ``cursor`` and
                     a ``complete`` flag.
``mode="range"``     spans ``start`` .. ``end`` (ordinals, end exclusive),
                     budget-limited and paginated by cursor.
``mode="metadata"``  entry shape: size, span counts by type, paths, symbols.

Spans are derived from the *exact stored original* (never from a rehydrated
marker): JSON rows, log blocks / tracebacks, search hits, code symbols, diff
hunks, or paragraphs. The item-span index is SQLite FTS5 (BM25) when the
interpreter's SQLite has it, otherwise a deterministic in-process BM25; the
semantic hybrid scorer and graph proximity are ranking hints only.

Safety: every read re-validates the parent entry in the requesting store (TTL
and deletion apply to spans with their parent; a changed original is a new
index generation), cursors are HMAC-bound to workspace, entry hash, index
generation, mode and query (no cross-entry leakage, no stale pagination), and
callers restrict selective modes to entries reachable in the requesting
conversation.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import math
import os
import re
import sqlite3
import threading
from collections import Counter, OrderedDict
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

MODES = ("full", "search", "range", "metadata")
DEFAULT_TOP_K = 5
MAX_TOP_K = 50
DEFAULT_MAX_TOKENS = 1200
MAX_INDEXED_ENTRIES = 256
MAX_SPAN_BYTES = 4000
LOG_BLOCK_LINES = 3

SPAN_TYPES = ("json_row", "log", "trace", "search_hit", "code_symbol", "diff_hunk", "paragraph")

_TERM_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./:\-]*[A-Za-z0-9_]|[A-Za-z0-9_]{2,}")
_STOP = frozenset(
    "the a an and or of to in on for is are was were be it this that with as at by from "
    "what why how when where which who does do did not no can should would".split()
)
_SEARCH_HIT_RE = re.compile(r"^((?:[A-Za-z]:)?[^\s:]+\.[A-Za-z0-9]{1,8}):(\d+)[:\-]")
_DIFF_FILE_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)")
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")
_CODE_DEF_RE = re.compile(
    r"^\s*(?:export\s+)?(?:pub(?:\([a-z]+\))?\s+)?(?:async\s+)?"
    r"(?:def|class|fn|func|function|struct|enum|trait|interface|impl|type)\s+([A-Za-z_][\w]*)"
)
_LOG_LINE_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}|\b(?:DEBUG|INFO|WARN|WARNING|ERROR|FATAL|TRACE)\b)"
)
_ID_KEYS = ("id", "key", "name", "path", "file", "uuid", "sha", "title")


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


def query_terms(query: str) -> list[str]:
    out: dict[str, None] = {}
    for m in _TERM_RE.finditer(query or ""):
        t = m.group(0).lower()
        if len(t) >= 2 and t not in _STOP:
            out.setdefault(t, None)
        if len(out) >= 32:
            break
    return list(out)


@dataclass(frozen=True)
class Span:
    entry_hash: str
    span_id: str
    ordinal: int
    byte_start: int
    byte_end: int
    type: str
    content_hash: str
    key: str = ""
    path: str = ""
    symbol: str = ""


# ─── Chunking (exact byte ranges over the stored original) ──────────────


class _Lines:
    """Line table with UTF-8 byte offsets."""

    def __init__(self, text: str) -> None:
        self.lines = text.splitlines(keepends=True)
        self.offsets: list[int] = []
        pos = 0
        for line in self.lines:
            self.offsets.append(pos)
            pos += len(line.encode("utf-8"))
        self.total = pos

    def span(self, start: int, end: int) -> tuple[int, int]:
        b0 = self.offsets[start]
        b1 = self.offsets[end] if end < len(self.lines) else self.total
        return b0, b1


def _classify(text: str, lines: list[str]) -> str:
    stripped = text.strip()
    if stripped[:1] in "[{":
        try:
            value = json.loads(stripped)
            if isinstance(value, list) or (
                isinstance(value, dict) and any(isinstance(v, list) for v in value.values())
            ):
                return "json"
        except ValueError:
            pass
    nonempty = [ln for ln in lines if ln.strip()]
    if not nonempty:
        return "paragraph"
    if any(_DIFF_FILE_RE.match(ln) for ln in nonempty[:50]) or (
        sum(1 for ln in nonempty if _HUNK_RE.match(ln)) >= 1
        and any(ln.startswith(("+++ ", "--- ")) for ln in nonempty)
    ):
        return "diff"
    hits = sum(1 for ln in nonempty if _SEARCH_HIT_RE.match(ln))
    if hits >= max(3, len(nonempty) // 2):
        return "search"
    defs = sum(1 for ln in nonempty if _CODE_DEF_RE.match(ln))
    logs = sum(1 for ln in nonempty if _LOG_LINE_RE.search(ln))
    if defs >= 2 and logs < len(nonempty) // 4:
        return "code"
    if logs >= max(3, len(nonempty) // 4) or any(ln.startswith("Traceback") for ln in nonempty):
        return "log"
    return "paragraph"


def _row_key(value: Any) -> str:
    if isinstance(value, dict):
        for k in _ID_KEYS:
            v = value.get(k)
            if isinstance(v, (str, int, float)) and not isinstance(v, bool):
                return f"{k}={v}"[:120]
    return ""


def _json_spans(text: str) -> list[tuple[int, int, str, str, str, str]] | None:
    """(byte_start, byte_end, type, key, path, symbol) per row of the largest array."""
    stripped_lead = len(text) - len(text.lstrip())
    try:
        root = json.loads(text)
    except ValueError:
        return None
    decoder = json.JSONDecoder()
    if isinstance(root, list):
        array_char_start = text.index("[", stripped_lead)
    else:
        # The largest top-level list value of an object.
        best_key, best_len = None, -1
        for k, v in root.items():
            if isinstance(v, list) and len(v) > best_len:
                best_key, best_len = k, len(v)
        if best_key is None:
            return None
        m = re.search(re.escape(json.dumps(best_key)) + r"\s*:\s*\[", text)
        if m is None:
            return None
        array_char_start = m.end() - 1
    # Walk the array with raw_decode to recover exact character offsets.
    prefix_bytes = _CharToByte(text)
    out: list[tuple[int, int, str, str, str, str]] = []
    pos = array_char_start + 1
    n = len(text)
    while pos < n:
        while pos < n and text[pos] in " \t\r\n,":
            pos += 1
        if pos >= n or text[pos] == "]":
            break
        try:
            value, end = decoder.raw_decode(text, pos)
        except ValueError:
            return None
        key = _row_key(value)
        path = ""
        if isinstance(value, dict):
            for pk in ("path", "file", "filename"):
                if isinstance(value.get(pk), str):
                    path = value[pk][:300]
                    break
        out.append((prefix_bytes(pos), prefix_bytes(end), "json_row", key, path, ""))
        pos = end
    return out or None


class _CharToByte:
    """Char offset -> UTF-8 byte offset (fast path for ASCII)."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.ascii = text.isascii()
        self._cache_pos = 0
        self._cache_bytes = 0

    def __call__(self, char_pos: int) -> int:
        if self.ascii:
            return char_pos
        if char_pos < self._cache_pos:
            self._cache_pos, self._cache_bytes = 0, 0
        self._cache_bytes += len(self.text[self._cache_pos : char_pos].encode("utf-8"))
        self._cache_pos = char_pos
        return self._cache_bytes


def chunk_original(entry_hash: str, original: str) -> list[Span]:
    """Semantic spans covering the original, in order, with exact byte ranges."""
    raw = original.encode("utf-8")
    lines = _Lines(original)
    shape = _classify(original, lines.lines)
    parts: list[tuple[int, int, str, str, str, str]] = []
    if shape == "json":
        parts = _json_spans(original) or []
        if not parts:
            shape = "paragraph"
    if shape == "diff":
        path = ""
        path_at_start = ""
        start = None
        for i, ln in enumerate(lines.lines):
            fm = _DIFF_FILE_RE.match(ln)
            if fm:
                path = fm.group(2)
            if fm or _HUNK_RE.match(ln):
                if start is not None:
                    b0, b1 = lines.span(start, i)
                    parts.append((b0, b1, "diff_hunk", "", path_at_start, ""))
                start, path_at_start = i, path
        if start is not None:
            b0, b1 = lines.span(start, len(lines.lines))
            parts.append((b0, b1, "diff_hunk", "", path_at_start, ""))
        if parts and parts[0][0] > 0:
            parts.insert(0, (0, parts[0][0], "diff_hunk", "", "", ""))
    elif shape == "search":
        for i, ln in enumerate(lines.lines):
            m = _SEARCH_HIT_RE.match(ln)
            b0, b1 = lines.span(i, i + 1)
            parts.append(
                (
                    b0,
                    b1,
                    "search_hit",
                    f"line={m.group(2)}" if m else "",
                    m.group(1) if m else "",
                    "",
                )
            )
    elif shape == "code":
        starts = [i for i, ln in enumerate(lines.lines) if _CODE_DEF_RE.match(ln)]
        if not starts or starts[0] != 0:
            starts.insert(0, 0)
        for j, s in enumerate(starts):
            e = starts[j + 1] if j + 1 < len(starts) else len(lines.lines)
            m = _CODE_DEF_RE.match(lines.lines[s])
            b0, b1 = lines.span(s, e)
            parts.append((b0, b1, "code_symbol", "", "", m.group(1) if m else ""))
    elif shape == "log":
        i = 0
        n = len(lines.lines)
        while i < n:
            if lines.lines[i].startswith("Traceback"):
                j = i + 1
                while j < n and (
                    lines.lines[j].startswith((" ", "\t")) or not lines.lines[j].strip()
                ):
                    j += 1
                j = min(n, j + 1)  # the exception line closes the block
                b0, b1 = lines.span(i, j)
                parts.append((b0, b1, "trace", "", "", ""))
                i = j
                continue
            j = i
            while j < n and j - i < LOG_BLOCK_LINES and not lines.lines[j].startswith("Traceback"):
                j += 1
                if j < n and not lines.lines[j - 1].strip():
                    break
            b0, b1 = lines.span(i, max(j, i + 1))
            parts.append((b0, b1, "log", "", "", ""))
            i = max(j, i + 1)
    elif shape == "paragraph":
        i = 0
        n = len(lines.lines)
        while i < n:
            j = i
            size = 0
            while j < n:
                size += len(lines.lines[j])
                j += 1
                if not lines.lines[j - 1].strip() or size >= MAX_SPAN_BYTES:
                    break
            b0, b1 = lines.span(i, j)
            parts.append((b0, b1, "paragraph", "", "", ""))
            i = j
    # Split oversized spans at line boundaries so every span stays selectable
    # under the token budget, and drop whitespace-only spans.
    spans: list[Span] = []
    for b0, b1, typ, key, path, symbol in parts:
        for c0, c1 in _split_bytes(raw, b0, b1):
            body = raw[c0:c1]
            if not body.strip():
                continue
            ordinal = len(spans)
            spans.append(
                Span(
                    entry_hash=entry_hash,
                    span_id=f"{entry_hash}:{ordinal}",
                    ordinal=ordinal,
                    byte_start=c0,
                    byte_end=c1,
                    type=typ,
                    content_hash=hashlib.sha256(body).hexdigest()[:16],
                    key=key,
                    path=path,
                    symbol=symbol,
                )
            )
    return spans


def _split_bytes(raw: bytes, b0: int, b1: int) -> Iterable[tuple[int, int]]:
    if b1 - b0 <= MAX_SPAN_BYTES:
        yield b0, b1
        return
    start = b0
    while start < b1:
        end = min(b1, start + MAX_SPAN_BYTES)
        if end < b1:
            nl = raw.rfind(b"\n", start, end)
            if nl > start:
                end = nl + 1
            else:
                # No newline: back off to a UTF-8 boundary.
                while end > start and (raw[end] & 0xC0) == 0x80:
                    end -= 1
        yield start, end
        start = end


# ─── Index ──────────────────────────────────────────────────────────────


def _fts5_available(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp._fts_probe USING fts5(x)")
        conn.execute("DROP TABLE IF EXISTS temp._fts_probe")
        return True
    except sqlite3.OperationalError:
        return False


class CursorError(ValueError):
    pass


class IndexedCCR:
    """Item-span index over CCR originals (process-local, derived data).

    The index lives in an in-process SQLite database: it is rebuilt lazily from
    the persistent store, so stateless mode and restarts need no extra file.
    """

    def __init__(self, *, use_fts: bool | None = None) -> None:
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE spans (entry_hash TEXT, ordinal INTEGER, span_id TEXT, "
            "byte_start INTEGER, byte_end INTEGER, type TEXT, content_hash TEXT, "
            "key TEXT, path TEXT, symbol TEXT, generation INTEGER, "
            "PRIMARY KEY (entry_hash, ordinal))"
        )
        self._conn.execute(
            "CREATE TABLE entries (entry_hash TEXT PRIMARY KEY, generation INTEGER, "
            "original_sha TEXT, byte_len INTEGER, span_count INTEGER)"
        )
        self.fts = _fts5_available(self._conn) if use_fts is None else use_fts
        if self.fts:
            self._conn.execute(
                "CREATE VIRTUAL TABLE spans_fts USING fts5(body, entry_hash UNINDEXED, "
                "ordinal UNINDEXED, tokenize=\"unicode61 tokenchars '_./:-'\")"
            )
        self._generation = 0
        self._lru: OrderedDict[str, None] = OrderedDict()
        self._secret = os.urandom(32)
        self._originals: dict[str, bytes] = {}

    # -- lifecycle ---------------------------------------------------------

    def purge(self, entry_hash: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM spans WHERE entry_hash = ?", (entry_hash,))
            self._conn.execute("DELETE FROM entries WHERE entry_hash = ?", (entry_hash,))
            if self.fts:
                self._conn.execute("DELETE FROM spans_fts WHERE entry_hash = ?", (entry_hash,))
            self._lru.pop(entry_hash, None)
            self._originals.pop(entry_hash, None)

    def clear(self) -> None:
        with self._lock:
            for h in list(self._lru):
                self.purge(h)

    def indexed_hashes(self) -> list[str]:
        with self._lock:
            return list(self._lru)

    def _original(self, store: Any, entry_hash: str) -> str | None:
        peek = getattr(store, "peek_original", None)
        if callable(peek):
            return peek(entry_hash)
        return None

    def ensure(self, store: Any, entry_hash: str) -> tuple[int, str] | None:
        """(generation, original) for a live entry, indexing it if needed.

        The parent entry is re-read from the requesting store on every call:
        a missing/expired entry purges its spans; a changed original (same hash
        key, different bytes) is re-indexed under a new generation.
        """
        original = self._original(store, entry_hash)
        if original is None:
            self.purge(entry_hash)
            return None
        sha = hashlib.sha256(original.encode("utf-8")).hexdigest()
        with self._lock:
            row = self._conn.execute(
                "SELECT generation, original_sha FROM entries WHERE entry_hash = ?",
                (entry_hash,),
            ).fetchone()
            if row is not None and row[1] == sha:
                self._lru.move_to_end(entry_hash)
                return int(row[0]), original
            if row is not None:
                self.purge(entry_hash)
            spans = chunk_original(entry_hash, original)
            self._generation += 1
            gen = self._generation
            raw = original.encode("utf-8")
            self._conn.executemany(
                "INSERT INTO spans VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        s.entry_hash,
                        s.ordinal,
                        s.span_id,
                        s.byte_start,
                        s.byte_end,
                        s.type,
                        s.content_hash,
                        s.key,
                        s.path,
                        s.symbol,
                        gen,
                    )
                    for s in spans
                ],
            )
            if self.fts:
                self._conn.executemany(
                    "INSERT INTO spans_fts (body, entry_hash, ordinal) VALUES (?,?,?)",
                    [
                        (
                            f"{raw[s.byte_start : s.byte_end].decode('utf-8', 'replace')} "
                            f"{s.path} {s.symbol} {s.key}",
                            s.entry_hash,
                            s.ordinal,
                        )
                        for s in spans
                    ],
                )
            self._conn.execute(
                "INSERT INTO entries VALUES (?,?,?,?,?)",
                (entry_hash, gen, sha, len(raw), len(spans)),
            )
            self._originals[entry_hash] = raw
            self._lru[entry_hash] = None
            while len(self._lru) > MAX_INDEXED_ENTRIES:
                old, _ = self._lru.popitem(last=False)
                self.purge(old)
            return gen, original

    def _spans(self, entry_hash: str) -> list[Span]:
        rows = self._conn.execute(
            "SELECT entry_hash, span_id, ordinal, byte_start, byte_end, type, content_hash, "
            "key, path, symbol FROM spans WHERE entry_hash = ? ORDER BY ordinal",
            (entry_hash,),
        ).fetchall()
        return [Span(*r) for r in rows]

    # -- cursors -----------------------------------------------------------

    def make_cursor(
        self,
        *,
        workspace: str,
        entry_hash: str,
        generation: int,
        mode: str,
        query: str,
        offset: int,
    ) -> str:
        payload = json.dumps(
            {
                "w": workspace,
                "h": entry_hash,
                "g": generation,
                "m": mode,
                "q": hashlib.sha256(query.encode("utf-8")).hexdigest()[:16],
                "o": offset,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        sig = hmac.new(self._secret, payload, hashlib.sha256).hexdigest()[:24]
        return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=") + "." + sig

    def read_cursor(
        self,
        cursor: str,
        *,
        workspace: str,
        entry_hash: str,
        generation: int,
        mode: str,
        query: str,
    ) -> int:
        try:
            body, sig = cursor.rsplit(".", 1)
            payload = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
        except (ValueError, TypeError) as exc:
            raise CursorError("cursor_invalid") from exc
        expected = hmac.new(self._secret, payload, hashlib.sha256).hexdigest()[:24]
        if not hmac.compare_digest(sig, expected):
            raise CursorError("cursor_invalid")
        data = json.loads(payload)
        if data.get("w") != workspace or data.get("h") != entry_hash or data.get("m") != mode:
            raise CursorError("cursor_scope_mismatch")
        if data.get("q") != hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]:
            raise CursorError("cursor_scope_mismatch")
        if data.get("g") != generation:
            raise CursorError("cursor_stale")
        return int(data.get("o", 0))

    # -- ranking -----------------------------------------------------------

    def _rank(
        self, entry_hash: str, spans: list[Span], query: str, hints: Collection[str]
    ) -> list[tuple[float, Span]]:
        terms = query_terms(query)
        if not terms:
            return []
        raw = self._originals.get(entry_hash, b"")
        texts = [raw[s.byte_start : s.byte_end].decode("utf-8", "replace") for s in spans]
        lexical: dict[int, float] = {}
        if self.fts:
            match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
            try:
                rows = self._conn.execute(
                    "SELECT ordinal, bm25(spans_fts) FROM spans_fts "
                    "WHERE spans_fts MATCH ? AND entry_hash = ? ORDER BY bm25(spans_fts) LIMIT 200",
                    (match, entry_hash),
                ).fetchall()
                # bm25() is lower-is-better (negative); flip to higher-is-better.
                lexical = {int(o): -float(b) for o, b in rows}
            except sqlite3.OperationalError:
                lexical = {}
        if not lexical:
            lexical = _lexical_bm25(terms, texts)
        if not lexical:
            return []
        top = max(lexical.values()) or 1.0
        exact = [t for t in terms if any(c in t for c in "_./:") or any(ch.isdigit() for ch in t)]
        candidates = sorted(lexical, key=lambda o: -lexical[o])[:64]
        semantic = _semantic_hints([texts[o] for o in candidates], query)
        hint_set = {h.lower() for h in hints}
        scored: list[tuple[float, Span]] = []
        for idx, o in enumerate(candidates):
            span = spans[o]
            score = lexical[o] / top
            low = texts[o].lower()
            if exact and any(t in low for t in exact):
                score += 0.25
            if semantic is not None:
                score = 0.75 * score + 0.25 * semantic[idx]
            if hint_set and (
                (span.path and span.path.lower() in hint_set)
                or (span.symbol and span.symbol.lower() in hint_set)
            ):
                score += 0.05  # graph proximity: a hint, never a gate
            scored.append((round(score, 6), span))
        scored.sort(key=lambda p: (-p[0], p[1].ordinal))
        return scored

    # -- public modes ------------------------------------------------------

    def _span_view(self, span: Span, text: str, score: float | None = None) -> dict[str, Any]:
        view: dict[str, Any] = {
            "span_id": span.span_id,
            "ordinal": span.ordinal,
            "type": span.type,
            "byte_start": span.byte_start,
            "byte_end": span.byte_end,
            "content_hash": span.content_hash,
            "text": text,
        }
        if score is not None:
            view["score"] = score
        for name in ("key", "path", "symbol"):
            if getattr(span, name):
                view[name] = getattr(span, name)
        return view

    def _take(
        self,
        entry_hash: str,
        ordered: list[tuple[float | None, Span]],
        offset: int,
        top_k: int,
        max_tokens: int,
    ) -> tuple[list[dict[str, Any]], int, int]:
        raw = self._originals.get(entry_hash, b"")
        out: list[dict[str, Any]] = []
        used = 0
        i = offset
        while i < len(ordered) and len(out) < top_k:
            score, span = ordered[i]
            text = raw[span.byte_start : span.byte_end].decode("utf-8", "replace")
            cost = estimate_tokens(text)
            if out and used + cost > max_tokens:
                break
            view = self._span_view(span, text, score)
            if cost > max_tokens:
                # A single span over the whole budget: return its head, flagged,
                # and point the caller at mode=range for the exact bytes.
                clipped = text.encode("utf-8")[: max_tokens * 4].decode("utf-8", "ignore")
                view["text"] = clipped
                view["span_truncated"] = True
                cost = estimate_tokens(clipped)
            out.append(view)
            used += cost
            i += 1
        return out, i, used

    def retrieve(
        self,
        store: Any,
        entry_hash: str,
        *,
        mode: str = "full",
        query: str = "",
        top_k: int | None = None,
        max_tokens: int | None = None,
        cursor: str | None = None,
        start: int | None = None,
        end: int | None = None,
        workspace: str = "",
        reachable: Collection[str] | None = None,
        hints: Collection[str] = (),
    ) -> dict[str, Any]:
        """Serve one retrieval. ``full`` is byte-exact; others are selective."""
        mode = (mode or "full").lower()
        if mode not in MODES:
            return {"hash": entry_hash, "error": f"unknown mode {mode!r}", "modes": list(MODES)}
        if mode != "full" and reachable is not None and entry_hash not in reachable:
            return {"hash": entry_hash, "error": "not_reachable_in_this_conversation"}
        state = self.ensure(store, entry_hash)
        if state is None:
            return {"hash": entry_hash, "error": "entry_not_found_or_expired", "status": "missing"}
        if mode != "metadata":
            # Unified-21 #12: a model re-retrieving content it was shown in a
            # lossy form is retention feedback for that decision.
            from headroom.agent.feedback import note_ccr_retrieval

            note_ccr_retrieval(entry_hash)
        generation, original = state
        if mode == "full":
            return {
                "hash": entry_hash,
                "mode": "full",
                "original_content": original,
                "complete": True,
            }
        k = max(1, min(MAX_TOP_K, int(top_k or DEFAULT_TOP_K)))
        budget = max(64, int(max_tokens or DEFAULT_MAX_TOKENS))
        with self._lock:
            spans = self._spans(entry_hash)
        base = {
            "hash": entry_hash,
            "source_hash": entry_hash,
            "mode": mode,
            "generation": generation,
            "total_spans": len(spans),
            "index": "fts5" if self.fts else "lexical",
        }
        if mode == "metadata":
            types = Counter(s.type for s in spans)
            return {
                **base,
                "byte_length": len(original.encode("utf-8")),
                "estimated_tokens": estimate_tokens(original),
                "span_types": dict(types),
                "paths": sorted({s.path for s in spans if s.path})[:50],
                "symbols": [s.symbol for s in spans if s.symbol][:50],
                "complete": True,
            }
        cursor_query = query if mode == "search" else f"{start}:{end}"
        try:
            offset = (
                self.read_cursor(
                    cursor,
                    workspace=workspace,
                    entry_hash=entry_hash,
                    generation=generation,
                    mode=mode,
                    query=cursor_query,
                )
                if cursor
                else 0
            )
        except CursorError as exc:
            return {**base, "error": str(exc)}
        if mode == "search":
            if not query.strip():
                return {**base, "error": "query required for mode=search"}
            ordered: list[tuple[float | None, Span]] = list(
                self._rank(entry_hash, spans, query, hints)
            )
        else:
            lo = max(0, int(start or 0))
            hi = min(len(spans), int(end) if end is not None else len(spans))
            ordered = [(None, s) for s in spans[lo:hi]]
        results, next_offset, used = self._take(entry_hash, ordered, offset, k, budget)
        has_more = next_offset < len(ordered)
        complete = (
            mode == "range"
            and offset == 0
            and not has_more
            and len(ordered) == len(spans)
            and not any(r.get("span_truncated") for r in results)
        )
        return {
            **base,
            "query": query if mode == "search" else None,
            "results": results,
            "returned_tokens": used,
            "matched_spans": len(ordered),
            "has_more": has_more,
            "cursor": (
                self.make_cursor(
                    workspace=workspace,
                    entry_hash=entry_hash,
                    generation=generation,
                    mode=mode,
                    query=cursor_query,
                    offset=next_offset,
                )
                if has_more
                else None
            ),
            "complete": complete,
        }


def _lexical_bm25(
    terms: list[str], texts: list[str], k1: float = 1.2, b: float = 0.75
) -> dict[int, float]:
    """Deterministic BM25 over spans when FTS5 is unavailable."""
    docs = [Counter(m.group(0).lower() for m in _TERM_RE.finditer(t)) for t in texts]
    n = len(docs)
    if n == 0:
        return {}
    avg = sum(sum(d.values()) for d in docs) / n or 1.0
    df = {t: sum(1 for d in docs if t in d) for t in terms}
    out: dict[int, float] = {}
    for i, d in enumerate(docs):
        dl = sum(d.values()) or 1
        score = 0.0
        for t in terms:
            tf = d.get(t, 0)
            if not tf:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg))
        if score > 0:
            out[i] = score
    return out


def _semantic_hints(texts: list[str], query: str) -> list[float] | None:
    """Rust hybrid relevance (BM25 + shared semantic model when loaded)."""
    if not texts:
        return None
    try:
        from headroom import _core

        if json.loads(_core.semantic_relevance_status()).get("state") != "ready":
            return None
        return list(_core.hybrid_relevance_scores([t[:2000] for t in texts], query))
    except Exception:  # noqa: BLE001 - hints are optional
        return None


_INDEX: IndexedCCR | None = None
_INDEX_LOCK = threading.Lock()


def get_indexed_ccr() -> IndexedCCR:
    global _INDEX
    if _INDEX is None:
        with _INDEX_LOCK:
            if _INDEX is None:
                _INDEX = IndexedCCR()
    return _INDEX


def reset_indexed_ccr() -> None:
    global _INDEX
    with _INDEX_LOCK:
        if _INDEX is not None:
            _INDEX.clear()
        _INDEX = None


def selective_args(args: dict[str, Any] | None) -> bool:
    """True when a retrieval asks for anything other than the legacy full read."""
    if not args:
        return False
    mode = str(args.get("mode") or "").lower()
    return bool((mode and mode != "full") or args.get("query"))


def retrieve_with_args(
    store: Any,
    entry_hash: str,
    args: dict[str, Any] | None,
    *,
    workspace: str = "",
    reachable: Collection[str] | None = None,
) -> dict[str, Any]:
    """Dispatch a parsed ``headroom_retrieve`` argument dict to the index."""
    args = args or {}
    mode = str(args.get("mode") or ("search" if args.get("query") else "full")).lower()

    def _int(name: str) -> int | None:
        v = args.get(name)
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    return get_indexed_ccr().retrieve(
        store,
        entry_hash,
        mode=mode,
        query=str(args.get("query") or ""),
        top_k=_int("top_k"),
        max_tokens=_int("max_tokens"),
        cursor=str(args["cursor"]) if args.get("cursor") else None,
        start=_int("start"),
        end=_int("end"),
        workspace=workspace,
        reachable=reachable,
    )

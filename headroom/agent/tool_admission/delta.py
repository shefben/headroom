"""Cross-turn semantic/structural deltas (unified-21 #5).

:class:`ResourceVersionStore` remembers, per conversation scope and stable
``resource_id``, the last *complete* version of a resource: its content hash,
the CCR hash of its exact original, its unit identities (line, keyed JSON
row, search row, listing entry or test outcome digests) and a completeness
flag. A partial version is recorded as partial and can never serve as a
complete comparison base.

:func:`compute_delta` compares the current complete result with that base
by the family's natural unit and renders a compact view with the base and
current hashes and precise added/removed/changed/unchanged counts. Removed
base content is shown only in bracketed lines (it is history, not a current
fact). An unchanged re-read becomes a small reference; the dispatcher emits
it only after the *current* exact original is stored and read back.

The store is bounded, in memory, and write-through to the agent ledger
(``tool_resource_versions``) when the agent runtime provides one: writes and
cold loads run on the runtime worker (``submit_task``), never on a request
thread.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from headroom.agent.interfaces import ToolResultEnvelope
from headroom.agent.tool_admission.families import ToolFamily
from headroom.agent.tool_admission.views import parse_test_outcomes

log = logging.getLogger(__name__)

MAX_UNITS = 20_000
MAX_RESOURCES = 2048
MAX_DELTA_LINES = 240
_ID_KEYS = ("id", "key", "uuid", "_id", "name", "path", "sha", "ref")

_TABLE = (
    "CREATE TABLE IF NOT EXISTS tool_resource_versions ("
    " scope TEXT NOT NULL, resource_id TEXT NOT NULL, version_hash TEXT NOT NULL,"
    " ccr_hash TEXT NOT NULL, complete INTEGER NOT NULL, family TEXT NOT NULL,"
    " unit_kind TEXT NOT NULL, units TEXT NOT NULL, total_units INTEGER NOT NULL,"
    " call_id TEXT NOT NULL, updated_at REAL NOT NULL, presented_exact INTEGER NOT NULL,"
    " PRIMARY KEY (scope, resource_id))"
)
_INDEX = (
    "CREATE INDEX IF NOT EXISTS tool_resource_versions_time"
    " ON tool_resource_versions(updated_at)"
)


def _unit_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:12]


@dataclass(frozen=True)
class ResourceVersion:
    resource_id: str
    version_hash: str
    ccr_hash: str
    complete: bool
    family: str
    unit_kind: str
    units: tuple[str, ...]
    total_units: int
    call_id: str = ""
    updated_at: float = field(default_factory=time.time)
    #: True when what the model saw of this version was exact (the original,
    #: a lossless form, or a delta over an exactly-presented base).
    presented_exact: bool = True


@dataclass(frozen=True)
class Delta:
    text: str
    label: str
    base_hash: str
    current_hash: str
    added: int
    removed: int
    changed: int
    unchanged: int
    unit: str
    reference_only: bool = False


# ---------------------------------------------------------------------------
# Units per family
# ---------------------------------------------------------------------------


def _json_rows(text: str) -> tuple[str, dict[str, str]] | None:
    """``(id_key, {row id: canonical row})`` for a keyed JSON array, else None."""
    try:
        value = json.loads(text)
    except ValueError:
        return None
    rows = value
    if isinstance(value, dict):
        lists = [v for v in value.values() if isinstance(v, list) and v and all(isinstance(r, dict) for r in v)]
        if len(lists) != 1:
            return None
        rows = lists[0]
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        return None
    for key in _ID_KEYS:
        ids = [r.get(key) for r in rows]
        if all(isinstance(i, (str, int)) for i in ids) and len(set(map(str, ids))) == len(ids):
            return key, {
                str(r[key]): json.dumps(r, sort_keys=True, separators=(",", ":")) for r in rows
            }
    return None


def units_for(family: ToolFamily, text: str) -> tuple[str, list[str]]:
    """``(unit_kind, ordered unit strings)`` for delta comparison."""
    if family is ToolFamily.JSON:
        rows = _json_rows(text)
        if rows is not None:
            key, mapping = rows
            return f"json_row:{key}", [f"{rid}\x1f{row}" for rid, row in mapping.items()]
    if family is ToolFamily.TEST:
        outcomes = parse_test_outcomes(text)
        if outcomes:
            return "test_outcome", [f"{tid}\x1f{o}" for tid, o in sorted(outcomes.items())]
    if family in (ToolFamily.SEARCH, ToolFamily.LISTING):
        kind = "search_row" if family is ToolFamily.SEARCH else "entry"
        return kind, [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    return "line", text.split("\n")


def _identity(unit_kind: str, unit: str) -> str:
    """Identity of a unit: keyed units by their key+value, others by text."""
    return _unit_digest(unit)


def version_for(
    envelope: ToolResultEnvelope, *, ccr_hash: str, family: ToolFamily
) -> ResourceVersion:
    kind, units = units_for(family, envelope.content)
    return ResourceVersion(
        resource_id=str(envelope.resource_id or ""),
        version_hash=envelope.version_hash,
        ccr_hash=ccr_hash,
        complete=envelope.is_complete and not envelope.stderr,
        family=family.value,
        unit_kind=kind,
        units=tuple(_identity(kind, u) for u in units[:MAX_UNITS]),
        total_units=len(units),
        call_id=envelope.tool_call_id,
    )


# ---------------------------------------------------------------------------
# Delta rendering
# ---------------------------------------------------------------------------


def _header(
    label: str, base: ResourceVersion, current: str, counts: str, ccr_hash: str
) -> str:
    return (
        f"[headroom delta: {label} vs earlier result; base version {base.version_hash[:12]}, "
        f"current version {current[:12]}; {counts}. "
        f"Exact current output: Retrieve more: hash={ccr_hash}]"
    )


def _bracket(text: str) -> str:
    return "[- " + text.replace("]", ")") + "]"


def compute_delta(
    family: ToolFamily,
    envelope: ToolResultEnvelope,
    base: ResourceVersion,
    *,
    base_text: str | None,
    ccr_hash: str,
    label: str,
) -> Delta | None:
    """Delta of a complete current result against a complete base (else None)."""
    if not base.complete or not envelope.is_complete or envelope.stderr:
        return None  # a partial base or current never qualifies as a comparison
    if base.family != family.value:
        return None
    current_hash = envelope.version_hash
    kind, units = units_for(family, envelope.content)
    if kind != base.unit_kind or len(units) > MAX_UNITS or base.total_units > MAX_UNITS:
        return None
    if current_hash == base.version_hash:
        text = (
            f"[headroom: unchanged since an earlier {label} in this conversation "
            f"(version {current_hash[:12]}, {len(units)} {kind.split(':')[0]} units, identical bytes). "
            f"Exact output: Retrieve more: hash={ccr_hash}]"
        )
        return Delta(
            text, "delta_unchanged", base.version_hash, current_hash, 0, 0, 0, len(units), kind, True
        )
    if kind == "line":
        return _line_delta(envelope, base, base_text, units, ccr_hash, label)
    return _set_delta(kind, envelope, base, base_text, family, units, ccr_hash, label)


def _line_delta(
    envelope: ToolResultEnvelope,
    base: ResourceVersion,
    base_text: str | None,
    units: list[str],
    ccr_hash: str,
    label: str,
) -> Delta | None:
    cur_ids = [_identity("line", u) for u in units]
    matcher = difflib.SequenceMatcher(a=list(base.units), b=cur_ids, autojunk=False)
    base_lines = base_text.split("\n") if base_text is not None else None
    if base_lines is not None and len(base_lines) != base.total_units:
        base_lines = None  # stored base does not match its recorded identity
    added = removed = changed = unchanged = 0
    body: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            unchanged += i2 - i1
            continue
        if tag == "replace":
            changed += min(i2 - i1, j2 - j1)
            removed += max(0, (i2 - i1) - (j2 - j1))
            added += max(0, (j2 - j1) - (i2 - i1))
        elif tag == "delete":
            removed += i2 - i1
        else:
            added += j2 - j1
        body.append(f"[@@ base lines {i1 + 1}-{i2} -> current lines {j1 + 1}-{j2} @@]")
        if base_lines is not None:
            body.extend(_bracket(base_lines[i]) for i in range(i1, i2))
        elif i2 > i1:
            body.append(f"[- {i2 - i1} base lines (base text no longer stored)]")
        body.extend(f"+ {units[j]}" for j in range(j1, j2))
        if len(body) > MAX_DELTA_LINES:
            return None  # too different: the family view is the better candidate
    counts = f"+{added} added, -{removed} removed, ~{changed} changed, {unchanged} unchanged lines omitted"
    text = _header(label, base, envelope.version_hash, counts, ccr_hash) + "\n" + "\n".join(body)
    return Delta(
        text, "delta_lines", base.version_hash, envelope.version_hash, added, removed, changed,
        unchanged, "line",
    )


def _set_delta(
    kind: str,
    envelope: ToolResultEnvelope,
    base: ResourceVersion,
    base_text: str | None,
    family: ToolFamily,
    units: list[str],
    ccr_hash: str,
    label: str,
) -> Delta | None:
    base_ids = set(base.units)
    cur_by_id = {_identity(kind, u): u for u in units}
    keyed = kind.startswith(("json_row", "test_outcome"))
    base_units: list[str] = units_for(family, base_text)[1] if base_text is not None else []
    added_units = [u for i, u in cur_by_id.items() if i not in base_ids]
    removed_ids = [i for i in base.units if i not in cur_by_id]
    removed_set = set(removed_ids)
    removed_units = [u for u in base_units if _identity(kind, u) in removed_set]
    changed: list[tuple[str, str]] = []
    if keyed:
        base_by_key = {u.split("\x1f", 1)[0]: u for u in base_units}
        cur_keys = {u.split("\x1f", 1)[0]: u for u in added_units}
        for k, u in cur_keys.items():
            if k in base_by_key:
                changed.append((base_by_key[k], u))
        changed_keys = {k for k in cur_keys if k in base_by_key}
        added_units = [u for u in added_units if u.split("\x1f", 1)[0] not in changed_keys]
        removed_units = [u for u in removed_units if u.split("\x1f", 1)[0] not in changed_keys]
    removed_count = max(len(removed_ids) - len(changed), 0)
    unchanged = len(units) - len(added_units) - len(changed)
    unit_name = {"search_row": "search rows", "entry": "entries", "test_outcome": "test outcomes"}.get(
        kind, "rows"
    )
    if len(added_units) + len(changed) * 2 + len(removed_units) > MAX_DELTA_LINES:
        return None

    def show(u: str) -> str:
        if kind == "test_outcome":
            tid, outcome = u.split("\x1f", 1)
            return f"{tid} {outcome}"
        return u.split("\x1f", 1)[1] if keyed else u

    body: list[str] = []
    if changed:
        body.append(f"-- changed {unit_name} (base -> current):")
        for old, new in changed:
            body.append(_bracket(show(old)))
            body.append("+ " + show(new))
    if added_units:
        body.append(f"-- added {unit_name}:")
        body.extend("+ " + show(u) for u in added_units)
    if removed_units:
        body.append(f"-- removed {unit_name}:")
        body.extend(_bracket(show(u)) for u in removed_units)
    elif removed_count:
        body.append(f"-- {removed_count} {unit_name} removed (base text no longer stored)")
    counts = (
        f"+{len(added_units)} added, -{removed_count} removed, ~{len(changed)} changed, "
        f"{unchanged} unchanged {unit_name} omitted"
    )
    text = _header(label, base, envelope.version_hash, counts, ccr_hash) + "\n" + "\n".join(body)
    return Delta(
        text,
        f"delta_{kind.split(':')[0]}",
        base.version_hash,
        envelope.version_hash,
        len(added_units),
        removed_count,
        len(changed),
        unchanged,
        kind,
    )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class ResourceVersionStore:
    """Bounded per-scope version memory with optional ledger write-through."""

    def __init__(self, max_resources: int = MAX_RESOURCES) -> None:
        self._lock = threading.Lock()
        self._mem: OrderedDict[tuple[str, str], ResourceVersion] = OrderedDict()
        self._max = max_resources
        self._db: Any = None
        self._submit: Callable[[Callable[[], Any]], bool] | None = None
        self._loaded_scopes: set[str] = set()
        self.counters = {"recorded": 0, "persisted": 0, "loaded": 0, "persist_errors": 0}

    # -- persistence binding ---------------------------------------------------

    def bind(self, ledger: Any, submit: Callable[[Callable[[], Any]], bool]) -> None:
        """Persist through ``ledger`` using the runtime worker ``submit``."""
        with ledger.lock:
            ledger.connection.execute(_TABLE)
            ledger.connection.execute(_INDEX)
        with self._lock:
            self._db = ledger
            self._submit = submit
            self._loaded_scopes.clear()

    def unbind(self) -> None:
        with self._lock:
            self._db = None
            self._submit = None
            self._loaded_scopes.clear()

    # -- API -------------------------------------------------------------------

    def get(self, scope: str, resource_id: str) -> ResourceVersion | None:
        with self._lock:
            record = self._mem.get((scope, resource_id))
            if record is not None:
                self._mem.move_to_end((scope, resource_id))
                return record
            need_load = self._db is not None and scope not in self._loaded_scopes
            if need_load:
                self._loaded_scopes.add(scope)
            submit, db = self._submit, self._db
        if need_load and submit is not None:
            submit(lambda: self._load_scope(db, scope))
        return None

    def put(self, scope: str, record: ResourceVersion) -> None:
        with self._lock:
            key = (scope, record.resource_id)
            self._mem[key] = record
            self._mem.move_to_end(key)
            while len(self._mem) > self._max:
                self._mem.popitem(last=False)
            self.counters["recorded"] += 1
            submit, db = self._submit, self._db
        if submit is not None and db is not None:
            submit(lambda: self._persist(db, scope, record))

    def clear(self) -> None:
        with self._lock:
            self._mem.clear()
            self._loaded_scopes.clear()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "resources": len(self._mem),
                "persistent": self._db is not None and bool(getattr(self._db, "persistent", False)),
                **self.counters,
            }

    # -- worker-side -----------------------------------------------------------

    def _persist(self, db: Any, scope: str, r: ResourceVersion) -> None:
        try:
            db.execute(
                "INSERT OR REPLACE INTO tool_resource_versions(scope, resource_id, version_hash,"
                " ccr_hash, complete, family, unit_kind, units, total_units, call_id, updated_at,"
                " presented_exact) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    scope,
                    r.resource_id,
                    r.version_hash,
                    r.ccr_hash,
                    int(r.complete),
                    r.family,
                    r.unit_kind,
                    ",".join(r.units),
                    r.total_units,
                    r.call_id,
                    r.updated_at,
                    int(r.presented_exact),
                ),
            )
            self.counters["persisted"] += 1
        except Exception as exc:  # noqa: BLE001 - persistence is best-effort
            self.counters["persist_errors"] += 1
            log.debug("tool resource version persist failed: %s", exc)

    def _load_scope(self, db: Any, scope: str) -> None:
        try:
            rows = db.query(
                "SELECT resource_id, version_hash, ccr_hash, complete, family, unit_kind, units,"
                " total_units, call_id, updated_at, presented_exact FROM tool_resource_versions"
                " WHERE scope=?"
                " ORDER BY updated_at DESC LIMIT ?",
                (scope, self._max),
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("tool resource version load failed: %s", exc)
            return
        with self._lock:
            for row in reversed(rows):
                key = (scope, str(row[0]))
                if key in self._mem:
                    continue  # a fresher in-memory record wins
                self._mem[key] = ResourceVersion(
                    resource_id=str(row[0]),
                    version_hash=str(row[1]),
                    ccr_hash=str(row[2]),
                    complete=bool(row[3]),
                    family=str(row[4]),
                    unit_kind=str(row[5]),
                    units=tuple(u for u in str(row[6]).split(",") if u),
                    total_units=int(row[7]),
                    call_id=str(row[8]),
                    updated_at=float(row[9]),
                    presented_exact=bool(row[10]),
                )
                self.counters["loaded"] += 1
            while len(self._mem) > self._max:
                self._mem.popitem(last=False)

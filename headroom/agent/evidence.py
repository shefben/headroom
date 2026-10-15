"""Provenance-aware Evidence Ledger (#17) — the ``AgentEvidenceStore`` owner.

An :class:`EvidenceFact` is one concrete, typed observation about the
workspace: an exit code, a test status, an error signature, a file's
existence or content hash, git state, a tool's availability, a port in use,
or an authoritative user statement. Raw tool text is untrusted *evidence*
only — it is never stored, never treated as instructions, and this is not CCR
raw storage, semantic memory or the savings ledger.

Rules implemented here:

* **Deterministic extraction** from structured events (:func:`extract_candidates`).
  Negative observations (missing file, permission denied, failing command,
  unavailable tool) are first-class facts. Credentials are scrubbed and every
  value is bounded; whole files are never persisted.
* **Normalized claim keys** (``predicate|subject``) so the same claim from
  different events corroborates instead of duplicating.
* **Freshness.** A newer observation supersedes an older observation of the
  same claim. Volatile facts (ports/processes) carry ``valid_until`` and turn
  ``STALE`` when not re-observed. A changed file supersedes facts bound to its
  old content. A mutating git operation stales prior git state.
* **Contradictions coexist.** An observation and a user statement that
  disagree are both labelled ``CONTRADICTED`` until a later observation or
  statement resolves them — incompatible booleans are never averaged.
* **Confidence = source type + corroboration**, never advisory certainty.
* **Provenance.** Every fact names the exact ``source_event_id`` and source
  digest; corroborating events are listed in ``metadata``. Every fact
  transition is also written as an ``EVIDENCE_CHANGE`` event, so the ledger can
  be rebuilt by replay (:meth:`AgentEvidenceStore.replay`).

Every read is scoped to one ``(workspace_id, session_id)``; there is no API
that lists another project's or user's facts.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

from headroom.agent.events import (
    AgentEvent,
    EventEnvelope,
    EventKind,
    Origin,
    content_digest,
    make_event,
)
from headroom.agent.text_safety import clip, safe_text

if TYPE_CHECKING:
    from headroom.agent.ledger_db import AgentLedgerDB

EVIDENCE_PROJECTOR = "evidence"
MAX_VALUE_JSON = 160
VOLATILE_TTL_S = 300.0
_TAIL_CHARS = 6000


class EvidenceStatus(str, Enum):
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    CONTRADICTED = "CONTRADICTED"
    STALE = "STALE"
    RETRACTED = "RETRACTED"


LIVE_STATUSES = (EvidenceStatus.ACTIVE.value, EvidenceStatus.CONTRADICTED.value)


class SourceKind(str, Enum):
    TEST = "test"
    TOOL_EXIT = "tool_exit"
    TOOL_TEXT = "tool_text"
    FILE = "file"
    GIT = "git"
    USER = "user"


OBSERVATIONAL = frozenset(
    {SourceKind.TEST, SourceKind.TOOL_EXIT, SourceKind.TOOL_TEXT, SourceKind.FILE, SourceKind.GIT}
)

BASE_CONFIDENCE: dict[SourceKind, float] = {
    SourceKind.TEST: 0.95,
    SourceKind.TOOL_EXIT: 0.95,
    SourceKind.FILE: 0.9,
    SourceKind.GIT: 0.9,
    SourceKind.USER: 0.9,
    SourceKind.TOOL_TEXT: 0.75,
}
CORROBORATION_STEP = 0.04
MAX_CONFIDENCE = 0.99


def normalize_subject(subject: str) -> str:
    text = " ".join(str(subject).replace("\\", "/").split())
    return text[:200]


def claim_key_for(predicate: str, subject: str) -> str:
    return f"{predicate.strip().lower()}|{normalize_subject(subject).lower()}"


def fact_id_for(workspace_id: str, session_id: str, claim_key: str, source_event_id: str) -> str:
    """Deterministic fact id; TaskState uses it to link atoms to evidence."""
    return (
        "ef_"
        + content_digest("\x1f".join((workspace_id, session_id, claim_key, source_event_id)))[:32]
    )


def _value_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    if len(encoded) <= MAX_VALUE_JSON:
        return encoded
    if isinstance(value, str):
        return json.dumps(clip(value, MAX_VALUE_JSON - 8))
    raise ValueError("evidence value exceeds the bounded size")


@dataclass(frozen=True)
class EvidenceFact:
    id: str
    claim_key: str
    subject: str
    predicate: str
    value_json: str
    polarity: str  # "positive" | "negative"
    source_kind: str
    source_event_id: str
    source_hash: str
    workspace_id: str
    session_id: str
    task_id: str | None
    confidence: float
    status: str
    valid_from: float
    valid_until: float | None
    supersedes_id: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    revision: int = 1

    @property
    def value(self) -> Any:
        return json.loads(self.value_json)

    def to_meta(self) -> dict[str, Any]:
        """Bounded scalar form carried by ``EVIDENCE_CHANGE`` events."""
        meta = dict(self.metadata)
        return {
            "fact_id": self.id,
            "claim_key": self.claim_key[:160],
            "subject": self.subject[:160],
            "predicate": self.predicate,
            "value_json": self.value_json,
            "polarity": self.polarity,
            "source_kind": self.source_kind,
            "source_event_id": self.source_event_id,
            "source_hash": self.source_hash,
            "task_id": self.task_id,
            "confidence": round(self.confidence, 4),
            "status": self.status,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "supersedes_id": self.supersedes_id,
            "fact_revision": self.revision,
            "conflicts": list(meta.get("conflicts") or [])[:8],
            "corroborated_by": list(meta.get("corroborated_by") or [])[:8],
            "content_bound": bool(meta.get("content_bound")),
            "resource_hash": meta.get("resource_hash"),
            "volatile": bool(meta.get("volatile")),
            "reason": meta.get("reason"),
        }

    @classmethod
    def from_meta(cls, workspace_id: str, session_id: str, meta: Mapping[str, Any]) -> EvidenceFact:
        metadata: dict[str, Any] = {
            "conflicts": list(meta.get("conflicts") or []),
            "corroborated_by": list(meta.get("corroborated_by") or []),
            "content_bound": bool(meta.get("content_bound")),
            "resource_hash": meta.get("resource_hash"),
            "volatile": bool(meta.get("volatile")),
        }
        if meta.get("reason"):
            metadata["reason"] = meta.get("reason")
        return cls(
            id=str(meta["fact_id"]),
            claim_key=str(meta["claim_key"]),
            subject=str(meta["subject"]),
            predicate=str(meta["predicate"]),
            value_json=str(meta["value_json"]),
            polarity=str(meta["polarity"]),
            source_kind=str(meta["source_kind"]),
            source_event_id=str(meta["source_event_id"]),
            source_hash=str(meta["source_hash"]),
            workspace_id=workspace_id,
            session_id=session_id,
            task_id=meta.get("task_id"),
            confidence=float(meta["confidence"]),
            status=str(meta["status"]),
            valid_from=float(meta["valid_from"]),
            valid_until=(
                float(meta["valid_until"]) if meta.get("valid_until") is not None else None
            ),
            supersedes_id=meta.get("supersedes_id"),
            metadata=metadata,
            revision=int(meta.get("fact_revision") or 1),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "claim_key": self.claim_key,
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "polarity": self.polarity,
            "source_kind": self.source_kind,
            "source_event_id": self.source_event_id,
            "source_hash": self.source_hash,
            "task_id": self.task_id,
            "confidence": self.confidence,
            "status": self.status,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "supersedes_id": self.supersedes_id,
            "metadata": dict(self.metadata),
            "revision": self.revision,
        }

    def compact(self) -> str:
        """One-line rendering for prompt state (no raw tool text)."""
        value = self.value
        if isinstance(value, dict):
            shown = ", ".join(f"{k}={v}" for k, v in sorted(value.items()))
        else:
            shown = json.dumps(value) if not isinstance(value, str) else value
        label = "" if self.status == EvidenceStatus.ACTIVE.value else f" [{self.status.lower()}]"
        return f"{self.predicate} {self.subject}: {shown}{label}"


@dataclass(frozen=True)
class FactCandidate:
    subject: str
    predicate: str
    value: Any
    polarity: str
    source_kind: SourceKind
    volatile: bool = False
    content_bound: bool = False
    resource_hash: str | None = None
    stales_content_of: str | None = None  # subject whose content-bound facts go stale
    stales_predicates: tuple[str, ...] = ()
    stales_claims: tuple[str, ...] = ()  # claim keys a success makes obsolete
    retracts_subject: str | None = None


# ---------------------------------------------------------------------------
# Deterministic extraction
# ---------------------------------------------------------------------------

_READ_TOOLS = {"read", "read_file", "view", "cat", "open_file", "readfile"}
_SHELL_BUILTINS = {"cd", "export", "set", "source", ".", "pushd", "popd", "env", "sudo"}
_NOT_FOUND = re.compile(
    r"(?:command not found|is not recognized as an internal or external command|"
    r"not recognized as the name of a cmdlet)",
    re.IGNORECASE,
)
_PERMISSION = re.compile(
    r"(?:permission denied|EACCES|access is denied|operation not permitted|EPERM\b)",
    re.IGNORECASE,
)
_MISSING = re.compile(
    r"(?:no such file or directory|ENOENT|file does not exist|does not exist|"
    r"cannot find the (?:file|path)|could not find file|FileNotFoundError)",
    re.IGNORECASE,
)
_ERROR_LINE = re.compile(
    r"^\s*(?:E\s+)?((?:[A-Za-z_][\w.]*)?(?:Error|Exception|Failure|Panic)\b(?::|\s).{0,300}|"
    r"(?:error|fatal|ERROR|FATAL)(?:\[[\w-]+\])?:\s*.{1,300}|"
    r"thread '.{1,80}' panicked at .{1,200}|"
    r"FAILED .{1,300})$",
    re.MULTILINE,
)
_PORT_IN_USE = re.compile(
    r"(?:address already in use|EADDRINUSE|only one usage of each socket address)"
    r"[^\n]*?(?::|port\s*)(\d{2,5})",
    re.IGNORECASE,
)
_PORT_LISTEN = re.compile(
    r"(?:listening|running|serving|started server) (?:on|at) [^\n]*?:(\d{2,5})\b",
    re.IGNORECASE,
)
_GIT_BRANCH = re.compile(r"^On branch (\S+)", re.MULTILINE)
_GIT_CLEAN = re.compile(r"nothing to commit,? (?:\(?working (?:tree|directory) clean)", re.I)
_GIT_PORCELAIN = re.compile(r"^\s?([MADRCU?!]{1,2})\s+(\S.*)$", re.MULTILINE)
_GIT_LONG = re.compile(
    r"^\s+(?:modified|new file|deleted|renamed|typechange):\s+(\S.*)$", re.MULTILINE
)
_USER_STATE = re.compile(
    r"\b([\w./\-]{2,80})\s+(?:is|are)\s+(not\s+)?"
    r"(installed|available|running|broken|working|deprecated|required|configured)\b",
    re.IGNORECASE,
)
_USER_MISSING = re.compile(
    r"\b(?:there is no|there's no)\s+([\w./\-]{2,80})|\b([\w./\-]{2,80})\s+does(?: not|n't) exist\b",
    re.IGNORECASE,
)
_USER_RETRACT = re.compile(
    r"\b(?:ignore|disregard|forget)\s+(?:what i said|my earlier (?:statement|comment)|that)"
    r"\s+about\s+([\w./\-]{2,80})",
    re.IGNORECASE,
)


def _command_subject(command: str) -> str:
    return clip(command, 120)


_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _command_word(command: str) -> str | None:
    """The executable a shell command runs (skipping env assignments and builtins)."""
    for segment in re.split(r"&&|\|\||;|\|", command):
        words = segment.strip().split()
        while words and _ENV_ASSIGN.match(words[0]):
            words.pop(0)
        if not words:
            continue
        word = words[0].strip("\"'")
        if word in _SHELL_BUILTINS:
            continue
        return word.replace("\\", "/").rsplit("/", 1)[-1][:60] or None
    return None


def _error_signature(text: str) -> str | None:
    tail = text[-_TAIL_CHARS:]
    matches = [m.group(1) for m in _ERROR_LINE.finditer(tail)]
    if not matches:
        return None
    return safe_text(matches[-1], 150)


def _git_state(text: str) -> list[FactCandidate]:
    out: list[FactCandidate] = []
    branch = _GIT_BRANCH.search(text)
    if branch:
        out.append(
            FactCandidate(
                "HEAD", "git.branch", clip(branch.group(1), 80), "positive", SourceKind.GIT
            )
        )
    paths: list[str] = []
    if _GIT_CLEAN.search(text):
        paths = []
    else:
        paths = [m.group(1).strip() for m in _GIT_LONG.finditer(text)]
        if not paths:
            paths = [m.group(2).strip() for m in _GIT_PORCELAIN.finditer(text)]
    if branch or _GIT_CLEAN.search(text) or paths:
        unique = sorted({p.replace("\\", "/") for p in paths})
        shown: list[str] = []
        for path in unique:
            trial = {"count": len(unique), "paths": [*shown, path]}
            if len(json.dumps(trial, separators=(",", ":"))) > MAX_VALUE_JSON:
                break
            shown.append(path)
        out.append(
            FactCandidate(
                "worktree",
                "git.changed_paths",
                {"count": len(unique), "paths": shown},
                "positive",
                SourceKind.GIT,
            )
        )
    return out


def extract_candidates(envelope: EventEnvelope) -> list[FactCandidate]:
    """Concrete facts derivable from one event (and its transient content)."""
    event = envelope.event
    meta = event.redacted_metadata
    text = envelope.content or ""
    kind = event.kind
    out: list[FactCandidate] = []

    if kind == EventKind.TEST_RESULT.value:
        command = str(meta.get("command_head") or meta.get("runner") or "tests")
        exit_code = meta.get("exit_code")
        passed = int(meta.get("passed") or 0)
        failed = int(meta.get("failed") or 0) + int(meta.get("errors") or 0)
        if meta.get("is_partial"):
            status = "inconclusive"
        elif exit_code == 0 or (exit_code is None and failed == 0 and passed > 0):
            status = "passed" if failed == 0 else "failed"
        elif (exit_code is not None and exit_code != 0) or failed > 0:
            status = "failed"
        else:
            return out
        value = {"status": status, "passed": passed, "failed": failed, "exit": exit_code}
        subject = _command_subject(command)
        out.append(
            FactCandidate(
                subject,
                "test.status",
                value,
                "positive" if status == "passed" else "negative",
                SourceKind.TEST,
                stales_claims=(
                    (claim_key_for("error.signature", subject),) if status == "passed" else ()
                ),
            )
        )
        return out

    if kind == EventKind.TOOL_RESULT.value:
        tool = str(meta.get("tool_name") or "")
        command = meta.get("command_head")
        path_subject = event.resource_id or (
            normalize_subject(str(meta["path"])) if meta.get("path") else None
        )
        exit_code = meta.get("exit_code")
        is_error = bool(meta.get("is_error"))
        partial = bool(meta.get("is_partial"))
        tail = text[-_TAIL_CHARS:]
        if command:
            subject = _command_subject(str(command))
            if exit_code is not None:
                out.append(
                    FactCandidate(
                        subject,
                        "command.exit",
                        int(exit_code),
                        "positive" if exit_code == 0 else "negative",
                        SourceKind.TOOL_EXIT,
                        stales_claims=(
                            (claim_key_for("error.signature", subject),) if exit_code == 0 else ()
                        ),
                    )
                )
            word = _command_word(str(command))
            if word:
                if _NOT_FOUND.search(tail):
                    out.append(
                        FactCandidate(
                            word, "tool.available", False, "negative", SourceKind.TOOL_TEXT
                        )
                    )
                elif exit_code == 0:
                    out.append(
                        FactCandidate(
                            word, "tool.available", True, "positive", SourceKind.TOOL_EXIT
                        )
                    )
        if text and (is_error or (exit_code is not None and exit_code != 0)):
            signature = _error_signature(text)
            if signature:
                subject = (
                    _command_subject(str(command))
                    if command
                    else (f"{tool}:{path_subject}" if path_subject else tool or "tool")
                )
                out.append(
                    FactCandidate(
                        subject,
                        "error.signature",
                        signature,
                        "negative",
                        SourceKind.TOOL_TEXT,
                        content_bound=path_subject is not None and not command,
                    )
                )
        if text and _PERMISSION.search(tail):
            out.append(
                FactCandidate(
                    path_subject or _command_subject(str(command or tool)),
                    "path.permission",
                    "denied",
                    "negative",
                    SourceKind.TOOL_TEXT,
                )
            )
        if text and path_subject and _MISSING.search(tail) and not _NOT_FOUND.search(tail):
            out.append(
                FactCandidate(path_subject, "file.exists", False, "negative", SourceKind.TOOL_TEXT)
            )
        elif path_subject and tool.lower() in _READ_TOOLS and not is_error and text:
            out.append(
                FactCandidate(path_subject, "file.exists", True, "positive", SourceKind.FILE)
            )
            if not partial and event.resource_hash:
                out.append(
                    FactCandidate(
                        path_subject,
                        "file.content_hash",
                        event.resource_hash[:16],
                        "positive",
                        SourceKind.FILE,
                        resource_hash=event.resource_hash[:16],
                        stales_content_of=path_subject,
                    )
                )
        if text and command and str(command).strip().startswith("git status"):
            out.extend(_git_state(text))
        if text:
            for match in _PORT_IN_USE.finditer(tail):
                out.append(
                    FactCandidate(
                        f"port:{match.group(1)}",
                        "port.in_use",
                        True,
                        "negative",
                        SourceKind.TOOL_TEXT,
                        volatile=True,
                    )
                )
            for match in _PORT_LISTEN.finditer(tail):
                out.append(
                    FactCandidate(
                        f"port:{match.group(1)}",
                        "port.in_use",
                        True,
                        "positive",
                        SourceKind.TOOL_TEXT,
                        volatile=True,
                    )
                )
        return _dedupe(out)

    if kind == EventKind.FILE_CHANGE.value:
        subject = event.resource_id or normalize_subject(str(meta.get("path") or ""))
        if subject:
            out.append(
                FactCandidate(
                    subject,
                    "file.modified",
                    True,
                    "positive",
                    SourceKind.FILE,
                    stales_content_of=subject,
                )
            )
            out.append(FactCandidate(subject, "file.exists", True, "positive", SourceKind.FILE))
        return out

    if kind == EventKind.GIT_CHANGE.value:
        operation = str(meta.get("operation") or "git")
        out.append(
            FactCandidate(
                operation,
                "git.operation",
                "applied",
                "positive",
                SourceKind.GIT,
                stales_predicates=("git.changed_paths", "git.branch"),
            )
        )
        return out

    if kind == EventKind.USER_INTENT.value and event.origin is Origin.USER and text:
        for match in _USER_RETRACT.finditer(text):
            out.append(
                FactCandidate(
                    match.group(1),
                    "user.retraction",
                    True,
                    "positive",
                    SourceKind.USER,
                    retracts_subject=match.group(1),
                )
            )
        for match in _USER_STATE.finditer(text):
            negated = bool(match.group(2))
            state = match.group(3).lower()
            out.append(
                FactCandidate(
                    match.group(1),
                    f"state.{state}",
                    not negated,
                    "negative" if negated else "positive",
                    SourceKind.USER,
                )
            )
        for match in _USER_MISSING.finditer(text):
            subject = match.group(1) or match.group(2)
            out.append(FactCandidate(subject, "file.exists", False, "negative", SourceKind.USER))
        return _dedupe(out)
    return out


def _dedupe(candidates: Iterable[FactCandidate]) -> list[FactCandidate]:
    seen: set[str] = set()
    out: list[FactCandidate] = []
    for cand in candidates:
        key = claim_key_for(cand.predicate, cand.subject)
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

_COLUMNS = (
    "id",
    "claim_key",
    "subject",
    "predicate",
    "value_json",
    "polarity",
    "source_kind",
    "source_event_id",
    "source_hash",
    "workspace_id",
    "session_id",
    "task_id",
    "confidence",
    "status",
    "valid_from",
    "valid_until",
    "supersedes_id",
    "metadata",
    "revision",
)


def _row_to_fact(row: Sequence[Any]) -> EvidenceFact:
    data = dict(zip(_COLUMNS, row, strict=True))
    return EvidenceFact(
        id=data["id"],
        claim_key=data["claim_key"],
        subject=data["subject"],
        predicate=data["predicate"],
        value_json=data["value_json"],
        polarity=data["polarity"],
        source_kind=data["source_kind"],
        source_event_id=data["source_event_id"],
        source_hash=data["source_hash"],
        workspace_id=data["workspace_id"],
        session_id=data["session_id"],
        task_id=data["task_id"],
        confidence=float(data["confidence"]),
        status=data["status"],
        valid_from=float(data["valid_from"]),
        valid_until=float(data["valid_until"]) if data["valid_until"] is not None else None,
        supersedes_id=data["supersedes_id"],
        metadata=json.loads(data["metadata"] or "{}"),
        revision=int(data["revision"]),
    )


def _same_class(a: str, b: str) -> bool:
    """Two sources whose disagreement is a change over time, not a conflict."""
    try:
        ka, kb = SourceKind(a), SourceKind(b)
    except ValueError:
        return a == b
    return ka == kb or (ka in OBSERVATIONAL and kb in OBSERVATIONAL)


class AgentEvidenceStore:
    """The single owner of evidence facts (see module docstring)."""

    def __init__(self, db: AgentLedgerDB, *, clock: Any = time.time) -> None:
        self.db = db
        self._clock = clock
        self.counters: dict[str, int] = {
            "facts_created": 0,
            "facts_corroborated": 0,
            "facts_superseded": 0,
            "facts_contradicted": 0,
            "facts_stale": 0,
            "facts_retracted": 0,
        }

    # -- writes -------------------------------------------------------------

    def _upsert(self, conn: Any, fact: EvidenceFact) -> None:
        row = (
            fact.id,
            fact.claim_key,
            fact.subject,
            fact.predicate,
            fact.value_json,
            fact.polarity,
            fact.source_kind,
            fact.source_event_id,
            fact.source_hash,
            fact.workspace_id,
            fact.session_id,
            fact.task_id,
            float(fact.confidence),
            fact.status,
            float(fact.valid_from),
            fact.valid_until,
            fact.supersedes_id,
            json.dumps(dict(fact.metadata), sort_keys=True, separators=(",", ":")),
            int(fact.revision),
        )
        conn.execute(
            f"INSERT INTO evidence_facts({', '.join(_COLUMNS)}, updated_at) "
            f"VALUES ({', '.join('?' for _ in _COLUMNS)}, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            + ", ".join(f"{c}=excluded.{c}" for c in _COLUMNS if c != "id")
            + ", updated_at=excluded.updated_at",
            (*row, self._clock()),
        )

    def _select(self, conn: Any, where: str, params: Sequence[Any]) -> list[EvidenceFact]:
        rows = conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM evidence_facts WHERE {where}", tuple(params)
        ).fetchall()
        return [_row_to_fact(r) for r in rows]

    def observe(
        self,
        event: AgentEvent,
        candidates: Sequence[FactCandidate],
        *,
        task_id: str | None = None,
    ) -> list[EvidenceFact]:
        """Apply candidates derived from ``event``; returns every changed fact."""
        if not candidates:
            return []
        now = self._clock()
        changed: dict[str, EvidenceFact] = {}
        ws, sess = event.workspace_id, event.session_id
        with self.db.lock:
            conn = self.db.connection
            conn.execute("BEGIN IMMEDIATE")
            try:
                inserted = conn.execute(
                    "INSERT OR IGNORE INTO evidence_processed(event_id, ts) VALUES (?, ?)",
                    (event.id, now),
                ).rowcount
                if inserted != 1:
                    conn.execute("COMMIT")
                    return []  # this source event was already applied (replay)
                for fact in self._expire(conn, ws, sess, now):
                    changed[fact.id] = fact
                for cand in candidates:
                    for fact in self._apply(conn, event, cand, task_id, now):
                        changed[fact.id] = fact
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return list(changed.values())

    def _set_status(
        self, conn: Any, fact: EvidenceFact, status: EvidenceStatus, **extra_meta: Any
    ) -> EvidenceFact:
        meta = dict(fact.metadata)
        meta.update({k: v for k, v in extra_meta.items() if v is not None})
        updated = replace(fact, status=status.value, metadata=meta, revision=fact.revision + 1)
        self._upsert(conn, updated)
        counter = {
            EvidenceStatus.SUPERSEDED: "facts_superseded",
            EvidenceStatus.CONTRADICTED: "facts_contradicted",
            EvidenceStatus.STALE: "facts_stale",
            EvidenceStatus.RETRACTED: "facts_retracted",
        }.get(status)
        if counter:
            self.counters[counter] += 1
        return updated

    def _apply(
        self,
        conn: Any,
        event: AgentEvent,
        cand: FactCandidate,
        task_id: str | None,
        now: float,
    ) -> list[EvidenceFact]:
        ws, sess = event.workspace_id, event.session_id
        changed: list[EvidenceFact] = []
        live = "status IN (?, ?)"
        # Invalidation side effects first (they never depend on the new value).
        if cand.stales_content_of:
            subject = normalize_subject(cand.stales_content_of)
            for fact in self._select(
                conn,
                f"workspace_id=? AND session_id=? AND (subject=? OR subject LIKE ?) AND {live}",
                (ws, sess, subject, "%:" + subject, *LIVE_STATUSES),
            ):
                bound = fact.metadata.get("content_bound") or fact.predicate == "file.content_hash"
                if bound and fact.metadata.get("resource_hash") != cand.resource_hash:
                    if fact.predicate == "file.content_hash" and cand.predicate == fact.predicate:
                        continue  # handled as a normal supersession below
                    changed.append(
                        self._set_status(conn, fact, EvidenceStatus.STALE, reason="content_changed")
                    )
        for predicate in cand.stales_predicates:
            for fact in self._select(
                conn,
                f"workspace_id=? AND session_id=? AND predicate=? AND {live}",
                (ws, sess, predicate, *LIVE_STATUSES),
            ):
                changed.append(
                    self._set_status(conn, fact, EvidenceStatus.STALE, reason="git_mutation")
                )
        for claim in cand.stales_claims:
            for fact in self._select(
                conn,
                f"workspace_id=? AND session_id=? AND claim_key=? AND {live}",
                (ws, sess, claim, *LIVE_STATUSES),
            ):
                changed.append(
                    self._set_status(conn, fact, EvidenceStatus.STALE, reason="later_success")
                )
        if cand.retracts_subject:
            subject = normalize_subject(cand.retracts_subject).lower()
            for fact in self._select(
                conn,
                f"workspace_id=? AND session_id=? AND source_kind=? AND {live}",
                (ws, sess, SourceKind.USER.value, *LIVE_STATUSES),
            ):
                if fact.subject.lower() == subject and fact.predicate != "user.retraction":
                    changed.append(
                        self._set_status(conn, fact, EvidenceStatus.RETRACTED, reason=event.id)
                    )
            return changed  # a retraction is an action, not a standing fact

        key = claim_key_for(cand.predicate, cand.subject)
        value_json = _value_json(cand.value)
        heads = self._select(
            conn,
            f"workspace_id=? AND session_id=? AND claim_key=? AND {live} ORDER BY valid_from",
            (ws, sess, key, *LIVE_STATUSES),
        )
        kind = cand.source_kind.value
        new_id = fact_id_for(ws, sess, key, event.id)
        if any(h.id == new_id for h in heads):
            return changed  # this exact event was already applied
        superseded_id: str | None = None
        survivors: list[EvidenceFact] = []
        for head in heads:
            differs = head.value_json != value_json or head.polarity != cand.polarity
            if differs and _same_class(head.source_kind, kind):
                changed.append(self._set_status(conn, head, EvidenceStatus.SUPERSEDED))
                superseded_id = head.id
            else:
                survivors.append(head)
        same = [h for h in survivors if h.value_json == value_json and h.polarity == cand.polarity]
        if same:
            target = same[-1]
            meta = dict(target.metadata)
            corroborated = [*list(meta.get("corroborated_by") or []), event.id][-8:]
            meta["corroborated_by"] = corroborated
            confidence = min(MAX_CONFIDENCE, target.confidence + CORROBORATION_STEP)
            valid_until = now + VOLATILE_TTL_S if cand.volatile else target.valid_until
            updated = replace(
                target,
                confidence=confidence,
                metadata=meta,
                valid_until=valid_until,
                revision=target.revision + 1,
            )
            self._upsert(conn, updated)
            self.counters["facts_corroborated"] += 1
            changed.append(updated)
            survivors = [updated if s.id == target.id else s for s in survivors]
        else:
            meta: dict[str, Any] = {
                "content_bound": cand.content_bound,
                "resource_hash": cand.resource_hash,
                "volatile": cand.volatile,
                "corroborated_by": [],
                "conflicts": [],
            }
            fact = EvidenceFact(
                id=new_id,
                claim_key=key,
                subject=normalize_subject(cand.subject),
                predicate=cand.predicate,
                value_json=value_json,
                polarity=cand.polarity,
                source_kind=kind,
                source_event_id=event.id,
                source_hash=event.payload_digest,
                workspace_id=ws,
                session_id=sess,
                task_id=task_id,
                confidence=BASE_CONFIDENCE.get(cand.source_kind, 0.7),
                status=EvidenceStatus.ACTIVE.value,
                valid_from=now,
                valid_until=now + VOLATILE_TTL_S if cand.volatile else None,
                supersedes_id=superseded_id,
                metadata=meta,
            )
            self._upsert(conn, fact)
            self.counters["facts_created"] += 1
            changed.append(fact)
            survivors.append(fact)
        changed.extend(self._label_conflicts(conn, survivors))
        return changed

    def _label_conflicts(self, conn: Any, heads: list[EvidenceFact]) -> list[EvidenceFact]:
        """CONTRADICTED while live heads disagree; ACTIVE again once they agree."""
        values = {(h.value_json, h.polarity) for h in heads}
        changed: list[EvidenceFact] = []
        for head in heads:
            if len(values) <= 1:
                if head.status != EvidenceStatus.ACTIVE.value:
                    meta = dict(head.metadata)
                    meta["conflicts"] = []
                    resolved = replace(
                        head,
                        status=EvidenceStatus.ACTIVE.value,
                        metadata=meta,
                        revision=head.revision + 1,
                    )
                    self._upsert(conn, resolved)
                    changed.append(resolved)
                continue
            others = [
                o.id
                for o in heads
                if (o.value_json, o.polarity) != (head.value_json, head.polarity)
            ]
            if head.status != EvidenceStatus.CONTRADICTED.value or sorted(
                head.metadata.get("conflicts") or []
            ) != sorted(others[-8:]):
                changed.append(
                    self._set_status(conn, head, EvidenceStatus.CONTRADICTED, conflicts=others[-8:])
                )
        return changed

    def _expire(self, conn: Any, ws: str, sess: str, now: float) -> list[EvidenceFact]:
        stale = self._select(
            conn,
            "workspace_id=? AND session_id=? AND status IN (?, ?) "
            "AND valid_until IS NOT NULL AND valid_until < ?",
            (ws, sess, *LIVE_STATUSES, now),
        )
        return [self._set_status(conn, f, EvidenceStatus.STALE, reason="expired") for f in stale]

    def apply_change(self, fact: EvidenceFact) -> bool:
        """Idempotently apply a replayed ``EVIDENCE_CHANGE`` (newer revision wins)."""
        with self.db.lock:
            conn = self.db.connection
            row = conn.execute(
                "SELECT revision FROM evidence_facts WHERE id=?", (fact.id,)
            ).fetchone()
            if row is not None and int(row[0]) >= fact.revision:
                return False
            conn.execute("BEGIN IMMEDIATE")
            try:
                self._upsert(conn, fact)
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return True

    def replay(self, events: Iterable[AgentEvent]) -> int:
        """Rebuild facts for every scope present in ``events`` from EVIDENCE_CHANGE events."""
        changes = [e for e in events if e.kind == EventKind.EVIDENCE_CHANGE.value]
        scopes = {(e.workspace_id, e.session_id) for e in changes}
        with self.db.lock:
            for ws, sess in scopes:
                self.db.connection.execute(
                    "DELETE FROM evidence_facts WHERE workspace_id=? AND session_id=?", (ws, sess)
                )
        applied = 0
        for event in changes:
            fact = EvidenceFact.from_meta(
                event.workspace_id, event.session_id, event.redacted_metadata
            )
            if self.apply_change(fact):
                applied += 1
        return applied

    # -- reads (always scoped) ----------------------------------------------

    def facts(
        self,
        workspace_id: str,
        session_id: str,
        *,
        statuses: Sequence[str] | None = None,
        claim_key: str | None = None,
        task_id: str | None = None,
        source_event_id: str | None = None,
        limit: int = 200,
    ) -> list[EvidenceFact]:
        clauses = ["workspace_id=?", "session_id=?"]
        params: list[Any] = [workspace_id, session_id]
        if statuses:
            clauses.append(f"status IN ({', '.join('?' for _ in statuses)})")
            params.extend(statuses)
        if claim_key is not None:
            clauses.append("claim_key=?")
            params.append(claim_key)
        if task_id is not None:
            clauses.append("task_id=?")
            params.append(task_id)
        if source_event_id is not None:
            clauses.append("source_event_id=?")
            params.append(source_event_id)
        sql = (
            f"SELECT {', '.join(_COLUMNS)} FROM evidence_facts WHERE {' AND '.join(clauses)} "
            f"ORDER BY valid_from, id LIMIT {max(1, int(limit))}"
        )
        return [_row_to_fact(r) for r in self.db.query(sql, params)]

    def get(self, workspace_id: str, session_id: str, fact_id: str) -> EvidenceFact | None:
        rows = self.db.query(
            f"SELECT {', '.join(_COLUMNS)} FROM evidence_facts "
            "WHERE workspace_id=? AND session_id=? AND id=?",
            (workspace_id, session_id, fact_id),
        )
        return _row_to_fact(rows[0]) if rows else None

    def heads(self, workspace_id: str, session_id: str, claim_key: str) -> list[EvidenceFact]:
        return self.facts(workspace_id, session_id, statuses=LIVE_STATUSES, claim_key=claim_key)

    def germane(
        self,
        workspace_id: str,
        session_id: str,
        *,
        task_id: str | None,
        resources: Sequence[str] = (),
        terms: Sequence[str] = (),
        limit: int = 8,
    ) -> list[EvidenceFact]:
        """Compact live facts relevant to the active task, most useful first.

        Germane = recorded during the active task, or about a resource the
        task touched, or naming a term from the task's atoms. Failures and
        conflicts rank before confirmations; then recency.
        """
        now = self._clock()
        live = self.facts(workspace_id, session_id, statuses=LIVE_STATUSES, limit=500)
        lowered_terms = {t.lower() for t in terms if len(t) >= 3}
        resource_set = {normalize_subject(r).lower() for r in resources}
        scored: list[tuple[tuple[int, int, float], EvidenceFact]] = []
        for fact in live:
            if fact.valid_until is not None and fact.valid_until < now:
                continue
            subject = fact.subject.lower()
            relevant = (
                (task_id is not None and fact.task_id == task_id)
                or subject in resource_set
                or any(term in subject for term in lowered_terms)
            )
            if not relevant:
                continue
            rank = (
                0 if fact.status == EvidenceStatus.CONTRADICTED.value else 1,
                0 if fact.polarity == "negative" else 1,
                -fact.valid_from,
            )
            scored.append((rank, fact))
        scored.sort(key=lambda item: item[0])
        return [fact for _, fact in scored[: max(0, limit)]]

    def stats(self) -> dict[str, Any]:
        return dict(self.counters)

    def status(self) -> dict[str, Any]:
        """Cheap counters for /health (no SQLite access)."""
        return {"counters": dict(self.counters)}


def change_event(fact: EvidenceFact) -> EventEnvelope:
    """The ``EVIDENCE_CHANGE`` audit/replay event for one fact transition."""
    meta = fact.to_meta()
    event = make_event(
        workspace_id=fact.workspace_id,
        session_id=fact.session_id,
        task_id=fact.task_id,
        kind=EventKind.EVIDENCE_CHANGE,
        origin=Origin.SYSTEM,
        source_id=fact.id,
        index=fact.revision,
        payload=meta,
        resource_id=fact.subject if fact.subject.startswith("path:") else None,
        metadata=meta,
    )
    return EventEnvelope(event=event)


class EvidenceProjector:
    """Projects raw events into facts and replays ``EVIDENCE_CHANGE`` events.

    Runs on the runtime worker thread. Raw events need their transient content
    (absent on replay), so replay rebuilds from the change events alone.
    """

    name = EVIDENCE_PROJECTOR

    def __init__(
        self,
        store: AgentEvidenceStore,
        *,
        emit: Any = None,
        active_task: Any = None,
    ) -> None:
        self.store = store
        self._emit = emit
        self._active_task = active_task or (lambda ws, sess: None)

    def project(self, envelope: EventEnvelope, tx: Any) -> None:
        event = envelope.event
        if event.kind == EventKind.EVIDENCE_CHANGE.value:
            self.store.apply_change(
                EvidenceFact.from_meta(
                    event.workspace_id, event.session_id, event.redacted_metadata
                )
            )
            return
        if envelope.content is None:
            return  # replay of a raw event: the EVIDENCE_CHANGE events carry its effect
        candidates = extract_candidates(envelope)
        if not candidates:
            return
        changed = self.store.observe(
            event, candidates, task_id=self._active_task(event.workspace_id, event.session_id)
        )
        if changed and self._emit is not None:
            self._emit([change_event(fact) for fact in changed])

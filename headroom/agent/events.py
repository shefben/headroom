"""Typed agent events: the provider-neutral record every agent feature consumes.

An :class:`AgentEvent` is a small, immutable fact about one wrapped session:
a user intent, a tool call, a tool result, a file/git/test observation, or a
decision taken by a later feature. Events carry **digests and redacted
metadata only** — never raw prompt or tool text. Exact originals stay in the
existing CCR store; the transient text a projector may need for the current
request travels beside the event in :class:`EventEnvelope` and is never
persisted.

Event identity is the idempotency key derived from ``(session, source id,
kind, index, content digest)``. The store inserts each key at most once, so a
streamed turn that is retried or replayed with the same history produces no
duplicate events and no repeated projection.

Kinds are versioned. The core kinds below are reserved now so later phases
register producers against stable names; a new kind is added with
:func:`register_event_kind` (name + version), never by rewriting the schema.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

EVENT_SCHEMA_VERSION = 1

# Redacted metadata limits: scalar values only, bounded key count and size.
_MAX_METADATA_KEYS = 24
_MAX_METADATA_STR = 160


class Origin(str, Enum):
    """Who authored the content an event describes."""

    USER = "user"  # an authentic, explicitly authored user turn
    HARNESS = "harness"  # client/harness-injected user-role content
    MODEL = "model"  # assistant output or a model-issued tool call
    TOOL = "tool"  # a tool result (untrusted evidence, never instructions)
    SYSTEM = "system"  # Headroom itself (decisions, state transitions)


class EventKind(str, Enum):
    """Core reserved event kinds (version 1)."""

    USER_INTENT = "USER_INTENT"
    MODEL_OUTPUT = "MODEL_OUTPUT"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    FILE_CHANGE = "FILE_CHANGE"
    GIT_CHANGE = "GIT_CHANGE"
    TEST_RESULT = "TEST_RESULT"
    TASK_STATE_CHANGE = "TASK_STATE_CHANGE"
    EVIDENCE_CHANGE = "EVIDENCE_CHANGE"
    SCOPE_DECISION = "SCOPE_DECISION"
    CONTRACT_DECISION = "CONTRACT_DECISION"
    VERIFICATION_DECISION = "VERIFICATION_DECISION"
    WORKFLOW_OBSERVED = "WORKFLOW_OBSERVED"
    WORKFLOW_EXECUTED = "WORKFLOW_EXECUTED"


_kind_lock = threading.Lock()
_KIND_VERSIONS: dict[str, int] = {kind.value: 1 for kind in EventKind}


def register_event_kind(name: str, version: int = 1) -> str:
    """Register a versioned event kind for a later producer.

    Names are upper-snake. Re-registering the same name at the same or a lower
    version is a no-op; a higher version supersedes (events keep the version
    they were written with in ``v``). Returns the canonical name.
    """
    canonical = name.strip().upper()
    if not canonical or not canonical.replace("_", "").isalnum():
        raise ValueError(f"invalid event kind {name!r}")
    if version < 1:
        raise ValueError("event kind version must be >= 1")
    with _kind_lock:
        if _KIND_VERSIONS.get(canonical, 0) < version:
            _KIND_VERSIONS[canonical] = version
    return canonical


def event_kind_version(name: str) -> int | None:
    """Registered version for a kind, or ``None`` when unknown."""
    with _kind_lock:
        return _KIND_VERSIONS.get(str(name).upper())


def registered_event_kinds() -> dict[str, int]:
    with _kind_lock:
        return dict(_KIND_VERSIONS)


def content_digest(value: Any) -> str:
    """Stable SHA-256 hex digest of text, bytes or JSON-able data."""
    if isinstance(value, bytes):
        data = value
    elif isinstance(value, str):
        data = value.encode("utf-8", "surrogatepass")
    else:
        data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8", "surrogatepass"
        )
    return hashlib.sha256(data).hexdigest()


def idempotency_key(session_id: str, source_id: str, kind: str, index: int, digest: str) -> str:
    """The stable event id: ``(session, source call/id, kind, index, digest)``."""
    raw = "\x1f".join((session_id, source_id, str(kind), str(int(index)), digest))
    return "ev_" + hashlib.sha256(raw.encode("utf-8", "surrogatepass")).hexdigest()[:40]


def redact_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep only bounded scalar metadata; strings are truncated, nested data dropped.

    Producers are expected to pass structural facts (tool name, byte length,
    exit code, flags). This is the last line of defence against a producer
    leaking content into persisted state.
    """
    out: dict[str, Any] = {}
    if not metadata:
        return out
    for key in sorted(metadata)[:_MAX_METADATA_KEYS]:
        value = metadata[key]
        name = str(key)[:64]
        if value is None or isinstance(value, bool | int | float):
            out[name] = value
        elif isinstance(value, str):
            out[name] = value[:_MAX_METADATA_STR]
        elif isinstance(value, list | tuple) and all(isinstance(v, str) for v in value):
            out[name] = [v[:_MAX_METADATA_STR] for v in list(value)[:8]]
    return out


@dataclass(frozen=True)
class AgentEvent:
    """One immutable, digest-only agent event."""

    v: int
    id: str
    workspace_id: str
    session_id: str
    task_id: str | None
    kind: str
    origin: Origin
    source_id: str
    resource_id: str | None
    resource_hash: str | None
    revision: int
    ts: float
    payload_digest: str
    redacted_metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "v": self.v,
            "id": self.id,
            "workspace_id": self.workspace_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "kind": self.kind,
            "origin": self.origin.value,
            "source_id": self.source_id,
            "resource_id": self.resource_id,
            "resource_hash": self.resource_hash,
            "revision": self.revision,
            "ts": self.ts,
            "payload_digest": self.payload_digest,
            "redacted_metadata": json.dumps(
                dict(self.redacted_metadata), sort_keys=True, separators=(",", ":")
            ),
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> AgentEvent:
        raw_meta = row.get("redacted_metadata") or "{}"
        meta = json.loads(raw_meta) if isinstance(raw_meta, str) else dict(raw_meta)
        return cls(
            v=int(row["v"]),
            id=str(row["id"]),
            workspace_id=str(row["workspace_id"]),
            session_id=str(row["session_id"]),
            task_id=row.get("task_id"),
            kind=str(row["kind"]),
            origin=Origin(str(row["origin"])),
            source_id=str(row["source_id"]),
            resource_id=row.get("resource_id"),
            resource_hash=row.get("resource_hash"),
            revision=int(row.get("revision") or 0),
            ts=float(row["ts"]),
            payload_digest=str(row["payload_digest"]),
            redacted_metadata=meta,
        )


def make_event(
    *,
    workspace_id: str,
    session_id: str,
    kind: str | EventKind,
    origin: Origin,
    source_id: str,
    index: int,
    payload: Any,
    task_id: str | None = None,
    resource_id: str | None = None,
    resource_hash: str | None = None,
    revision: int = 0,
    metadata: Mapping[str, Any] | None = None,
    ts: float | None = None,
) -> AgentEvent:
    """Build an event; ``payload`` is digested and never stored."""
    kind_name = kind.value if isinstance(kind, EventKind) else register_event_kind(str(kind))
    version = event_kind_version(kind_name) or EVENT_SCHEMA_VERSION
    digest = content_digest(payload)
    return AgentEvent(
        v=version,
        id=idempotency_key(session_id, source_id, kind_name, index, digest),
        workspace_id=workspace_id,
        session_id=session_id,
        task_id=task_id,
        kind=kind_name,
        origin=origin,
        source_id=source_id,
        resource_id=resource_id,
        resource_hash=resource_hash,
        revision=int(revision),
        ts=time.time() if ts is None else float(ts),
        payload_digest=digest,
        redacted_metadata=redact_metadata(metadata),
    )


@dataclass(frozen=True)
class EventEnvelope:
    """An event plus transient, request-scoped content for in-process projectors.

    ``content`` is the normalized text the event was derived from (a user
    turn's authored text, a tool result's text). It lives only for the
    duration of projection and is never written to the store.
    """

    event: AgentEvent
    content: str | None = None

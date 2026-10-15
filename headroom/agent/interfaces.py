"""Small versioned interfaces that later agent phases consume, never redefine.

* :class:`TaskContext` — the request facts one wrapped turn exposes to request
  stages (identity, provider, the events this request produced).
* :class:`DecisionHints` — the compact, JSON-able hint structure passed from
  Python to the Rust live-zone functions (and produced by the JevK5 advisor).
* :class:`ToolResultEnvelope` — a structural description of one tool result,
  including truncation/partiality, so no consumer treats a cut-off result as
  complete evidence.
* :class:`ActionAuthorization` — a revision-pinned, expiring authorization for
  a path-sensitive action. Only a trusted wrap-registered identity can obtain
  one; a model or JevK5 confidence score never can.

Every type carries ``v`` and refuses to deserialize an unknown major version.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

TASK_CONTEXT_VERSION = 1
DECISION_HINTS_VERSION = 1
TOOL_RESULT_ENVELOPE_VERSION = 1
ACTION_AUTHORIZATION_VERSION = 1


class InterfaceVersionError(ValueError):
    """A serialized interface value has an unsupported version."""


def _check_version(value: Mapping[str, Any], expected: int, what: str) -> None:
    if int(value.get("v", -1)) != expected:
        raise InterfaceVersionError(f"unsupported {what} version {value.get('v')!r}")


@dataclass(frozen=True)
class TaskContext:
    """Request-scoped facts for one wrapped turn."""

    v: int
    request_id: str
    workspace_id: str
    session_id: str
    trusted: bool
    provider: str
    model: str
    tool_names: tuple[str, ...] = ()
    event_ids: tuple[str, ...] = ()
    latest_user_digest: str | None = None
    hints: DecisionHints | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tool_names"] = list(self.tool_names)
        data["event_ids"] = list(self.event_ids)
        data["hints"] = self.hints.to_dict() if self.hints else None
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TaskContext:
        _check_version(value, TASK_CONTEXT_VERSION, "TaskContext")
        hints = value.get("hints")
        return cls(
            v=TASK_CONTEXT_VERSION,
            request_id=str(value["request_id"]),
            workspace_id=str(value["workspace_id"]),
            session_id=str(value["session_id"]),
            trusted=bool(value["trusted"]),
            provider=str(value["provider"]),
            model=str(value["model"]),
            tool_names=tuple(value.get("tool_names") or ()),
            event_ids=tuple(value.get("event_ids") or ()),
            latest_user_digest=value.get("latest_user_digest"),
            hints=DecisionHints.from_dict(hints) if hints else None,
        )


@dataclass(frozen=True)
class DecisionHints:
    """Compact advisory hints for downstream transforms (Python or Rust FFI).

    ``advisory`` maps an option label to its probability. ``weight`` is the
    utility weight a consumer may give the advisory (already capped by the
    advisor's configured cap); deterministic hints carry weight 0.
    """

    v: int = DECISION_HINTS_VERSION
    family: str = ""
    source: str = "deterministic"  # "deterministic" | "jevk5"
    query_terms: tuple[str, ...] = ()
    advisory: Mapping[str, float] = field(default_factory=dict)
    choice: str | None = None
    confidence: float = 0.0
    weight: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "v": self.v,
            "family": self.family,
            "source": self.source,
            "query_terms": list(self.query_terms),
            "advisory": {str(k): float(p) for k, p in self.advisory.items()},
            "choice": self.choice,
            "confidence": float(self.confidence),
            "weight": float(self.weight),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DecisionHints:
        _check_version(value, DECISION_HINTS_VERSION, "DecisionHints")
        return cls(
            family=str(value.get("family") or ""),
            source=str(value.get("source") or "deterministic"),
            query_terms=tuple(str(t) for t in value.get("query_terms") or ()),
            advisory={str(k): float(p) for k, p in (value.get("advisory") or {}).items()},
            choice=value.get("choice"),
            confidence=float(value.get("confidence") or 0.0),
            weight=float(value.get("weight") or 0.0),
        )


_TRUNCATION_MARKERS = (
    re.compile(r"\[(?:output|content|result)?\s*truncated", re.IGNORECASE),
    re.compile(r"\.\.\.\s*\(\s*\d+\s+(?:more\s+)?(?:lines?|chars?|characters|bytes)", re.I),
    re.compile(r"<truncated", re.IGNORECASE),
    re.compile(r"output (?:was|has been) truncated", re.IGNORECASE),
    re.compile(r"\[\d+ (?:lines?|items?) (?:omitted|elided|hidden)", re.IGNORECASE),
    re.compile(r"Retrieve more: hash=", re.IGNORECASE),
)
_EXIT_CODE = re.compile(r"(?:exit(?:ed)?(?: with)?(?: code| status)?[:= ]+|Exit code:? )(-?\d{1,3})\b")


@dataclass(frozen=True)
class ToolResultEnvelope:
    """Structural description of one tool result (no content)."""

    v: int
    tool_call_id: str
    tool_name: str
    byte_length: int
    line_count: int
    is_error: bool
    is_partial: bool
    partial_reason: str | None
    source_truncated: bool
    source_byte_length: int | None
    exit_code: int | None

    @classmethod
    def from_text(
        cls,
        *,
        tool_call_id: str,
        tool_name: str,
        text: str,
        is_error: bool = False,
        source_byte_length: int | None = None,
    ) -> ToolResultEnvelope:
        """Derive the envelope deterministically from the result text."""
        raw = text.encode("utf-8", "replace")
        partial_reason = None
        for marker in _TRUNCATION_MARKERS:
            if marker.search(text):
                partial_reason = "truncation_marker"
                break
        source_truncated = source_byte_length is not None and source_byte_length > len(raw)
        if source_truncated and partial_reason is None:
            partial_reason = "source_larger_than_result"
        exit_code = None
        match = _EXIT_CODE.search(text[-4000:])
        if match:
            try:
                exit_code = int(match.group(1))
            except ValueError:
                exit_code = None
        return cls(
            v=TOOL_RESULT_ENVELOPE_VERSION,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            byte_length=len(raw),
            line_count=text.count("\n") + (1 if text else 0),
            is_error=bool(is_error) or (exit_code is not None and exit_code != 0),
            is_partial=partial_reason is not None,
            partial_reason=partial_reason,
            source_truncated=source_truncated,
            source_byte_length=source_byte_length,
            exit_code=exit_code,
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "envelope_v": self.v,
            "tool_name": self.tool_name,
            "byte_length": self.byte_length,
            "line_count": self.line_count,
            "is_error": self.is_error,
            "is_partial": self.is_partial,
            "partial_reason": self.partial_reason,
            "source_truncated": self.source_truncated,
            "source_byte_length": self.source_byte_length,
            "exit_code": self.exit_code,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ToolResultEnvelope:
        _check_version(value, TOOL_RESULT_ENVELOPE_VERSION, "ToolResultEnvelope")
        return cls(**{k: value[k] for k in cls.__dataclass_fields__})


class AuthorizationDecision(str, Enum):
    ALLOW = "allow"
    WARN = "warn"
    DENY = "deny"


@dataclass(frozen=True)
class ActionAuthorization:
    """Revision-pinned, expiring authorization for one path-sensitive action."""

    v: int
    workspace_id: str
    session_id: str
    task_revision: int
    scope_revision: int
    resource_revision: int
    expires_at: float
    decision: AuthorizationDecision

    def is_valid(
        self,
        *,
        workspace_id: str,
        session_id: str,
        task_revision: int,
        scope_revision: int,
        resource_revision: int,
        now: float | None = None,
    ) -> bool:
        """True only for an unexpired ALLOW pinned to exactly these revisions."""
        current = time.time() if now is None else now
        return (
            self.decision is AuthorizationDecision.ALLOW
            and current < self.expires_at
            and self.workspace_id == workspace_id
            and self.session_id == session_id
            and self.task_revision == task_revision
            and self.scope_revision == scope_revision
            and self.resource_revision == resource_revision
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["decision"] = self.decision.value
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ActionAuthorization:
        _check_version(value, ACTION_AUTHORIZATION_VERSION, "ActionAuthorization")
        return cls(
            v=ACTION_AUTHORIZATION_VERSION,
            workspace_id=str(value["workspace_id"]),
            session_id=str(value["session_id"]),
            task_revision=int(value["task_revision"]),
            scope_revision=int(value["scope_revision"]),
            resource_revision=int(value["resource_revision"]),
            expires_at=float(value["expires_at"]),
            decision=AuthorizationDecision(str(value["decision"])),
        )

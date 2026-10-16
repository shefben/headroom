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

import hashlib
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

TASK_CONTEXT_VERSION = 1
RELEVANCE_QUERY_VERSION = 1
RELEVANCE_QUERY_MAX_TOKENS = 512
DECISION_HINTS_VERSION = 1
TOOL_RESULT_ENVELOPE_VERSION = 2

#: Headroom's own workflow tool (unified-21 #20): its steps are ingested as
#: individual tool events, so its aggregate result is never re-ingested.
WORKFLOW_TOOL_NAMES = frozenset({"headroom_workflow", "mcp__headroom__headroom_workflow"})
#: v1 envelopes (no admission fields) still deserialize; their new fields default.
_ENVELOPE_READABLE_VERSIONS = (1, 2)
ACTION_AUTHORIZATION_VERSION = 1


class InterfaceVersionError(ValueError):
    """A serialized interface value has an unsupported version."""


def _check_version(value: Mapping[str, Any], expected: int, what: str) -> None:
    if int(value.get("v", -1)) != expected:
        raise InterfaceVersionError(f"unsupported {what} version {value.get('v')!r}")


def estimate_query_tokens(text: str) -> int:
    """Conservative token estimate (~3 UTF-8 bytes per token) for hard query caps."""
    return (len(text.encode("utf-8")) + 2) // 3


def _fit_terms(values: tuple[str, ...], budget: int) -> str:
    joined = ""
    for value in values:
        trial = f"{joined} {value}".strip()
        if estimate_query_tokens(trial) + 1 > budget:
            break
        joined = trial
    return joined


@dataclass(frozen=True)
class RelevanceQuery:
    """The compact, typed relevance query every downstream consumer shares.

    ``explicit_user_goal + tool_intent + exact_entities + active_error +
    top_graph_neighbors`` (graph only for code tasks). Exact user-named ids
    and paths are budgeted before prose; :attr:`text` never exceeds
    :data:`RELEVANCE_QUERY_MAX_TOKENS`. No conversation dump and no generated
    "thoughts" — only the latest authentic user goal and observed facts.
    """

    v: int = RELEVANCE_QUERY_VERSION
    explicit_user_goal: str = ""
    tool_intent: str = ""
    exact_entities: tuple[str, ...] = ()
    active_error: str = ""
    graph_neighbors: tuple[str, ...] = ()
    is_code: bool = False
    max_tokens: int = RELEVANCE_QUERY_MAX_TOKENS

    @property
    def text(self) -> str:
        """Query text in spec order, with the budget spent by priority."""
        budget = self.max_tokens
        chosen: dict[str, str] = {}

        def take(name: str, value: str) -> None:
            nonlocal budget
            if not value or budget <= 1:
                return
            cost = estimate_query_tokens(value) + 1
            if cost > budget:
                raw = value.encode("utf-8")[: max(0, (budget - 1) * 3)]
                value = raw.decode("utf-8", "ignore")
                if " " in value:
                    value = value.rsplit(" ", 1)[0]
                cost = estimate_query_tokens(value) + 1
            if value and cost <= budget:
                chosen[name] = value
                budget -= cost

        # Priority: exact entities > active error > goal prose > intent > graph.
        take("entities", _fit_terms(self.exact_entities, budget))
        take("error", self.active_error)
        take("goal", self.explicit_user_goal)
        take("intent", self.tool_intent)
        if self.is_code and self.graph_neighbors:
            take("graph", _fit_terms(self.graph_neighbors, budget))
        order = ("goal", "intent", "entities", "error", "graph")
        return "\n".join(chosen[name] for name in order if name in chosen)

    @property
    def terms(self) -> tuple[str, ...]:
        """Distinct query terms, exact entities first (for lexical/BM25 consumers)."""
        out: dict[str, None] = {}
        for entity in self.exact_entities:
            out[entity] = None
        for word in re.findall(r"[A-Za-z_][A-Za-z0-9_./-]{2,}", self.text):
            out.setdefault(word, None)
        return tuple(list(out)[:64])

    @property
    def hints(self) -> DecisionHints | None:
        """Deterministic relevance hints for the Rust live zone (FFI ``"hints"``).

        The query terms the token-budgeted :attr:`text` could not carry —
        typically exact entities past the budget. Relevance-only on the Rust
        side; ``None`` when the text already holds every term.
        """
        text = self.text.lower()
        extra = tuple(t for t in self.terms if t.lower() not in text)
        if not extra:
            return None
        return DecisionHints(family="relevance_query", query_terms=extra)

    def to_dict(self) -> dict[str, Any]:
        hints = self.hints
        return {
            "v": self.v,
            "explicit_user_goal": self.explicit_user_goal,
            "tool_intent": self.tool_intent,
            "exact_entities": list(self.exact_entities),
            "active_error": self.active_error,
            "graph_neighbors": list(self.graph_neighbors),
            "is_code": self.is_code,
            "max_tokens": self.max_tokens,
            "text": self.text,
            "hints": hints.to_dict() if hints is not None else None,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RelevanceQuery:
        _check_version(value, RELEVANCE_QUERY_VERSION, "RelevanceQuery")
        return cls(
            explicit_user_goal=str(value.get("explicit_user_goal") or ""),
            tool_intent=str(value.get("tool_intent") or ""),
            exact_entities=tuple(str(e) for e in value.get("exact_entities") or ()),
            active_error=str(value.get("active_error") or ""),
            graph_neighbors=tuple(str(n) for n in value.get("graph_neighbors") or ()),
            is_code=bool(value.get("is_code")),
            max_tokens=int(value.get("max_tokens") or RELEVANCE_QUERY_MAX_TOKENS),
        )


_TUPLE_FIELDS = (
    "tool_names",
    "event_ids",
    "exact_paths",
    "exact_symbols",
    "exact_errors",
    "exact_identifiers",
    "graph_neighbors",
)


@dataclass(frozen=True)
class TaskContext:
    """Request-scoped facts for one wrapped turn (shared by every consumer)."""

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
    # -- Phase 2 query model (optional, request-scoped, never persisted) -----
    latest_user_goal: str | None = None
    tool_intent: str | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    exact_paths: tuple[str, ...] = ()
    exact_symbols: tuple[str, ...] = ()
    exact_errors: tuple[str, ...] = ()
    exact_identifiers: tuple[str, ...] = ()
    content_type: str | None = None
    result_complete: bool | None = None
    result_truncated: bool = False
    resource_id: str | None = None
    resource_hash: str | None = None
    cache_policy: str | None = None
    graph_neighbors: tuple[str, ...] = ()
    task_id: str | None = None
    task_revision: int = 0
    relevance_query: RelevanceQuery | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for name in _TUPLE_FIELDS:
            data[name] = list(getattr(self, name))
        data["hints"] = self.hints.to_dict() if self.hints else None
        data["relevance_query"] = self.relevance_query.to_dict() if self.relevance_query else None
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
            latest_user_goal=value.get("latest_user_goal"),
            tool_intent=value.get("tool_intent"),
            tool_name=value.get("tool_name"),
            tool_call_id=value.get("tool_call_id"),
            exact_paths=tuple(value.get("exact_paths") or ()),
            exact_symbols=tuple(value.get("exact_symbols") or ()),
            exact_errors=tuple(value.get("exact_errors") or ()),
            exact_identifiers=tuple(value.get("exact_identifiers") or ()),
            content_type=value.get("content_type"),
            result_complete=value.get("result_complete"),
            result_truncated=bool(value.get("result_truncated")),
            resource_id=value.get("resource_id"),
            resource_hash=value.get("resource_hash"),
            cache_policy=value.get("cache_policy"),
            graph_neighbors=tuple(value.get("graph_neighbors") or ()),
            task_id=value.get("task_id"),
            task_revision=int(value.get("task_revision") or 0),
            relevance_query=(
                RelevanceQuery.from_dict(value["relevance_query"])
                if value.get("relevance_query")
                else None
            ),
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


def content_version_hash(text: str, stderr: str = "") -> str:
    """Version hash of a tool result's content (stdout and stderr kept distinct)."""
    raw = text.encode("utf-8", "replace")
    return hashlib.sha256(
        len(raw).to_bytes(8, "big") + raw + stderr.encode("utf-8", "replace")
    ).hexdigest()[:24]


@dataclass(frozen=True)
class ToolResultEnvelope:
    """One observed tool result, normalized (unified-21 #4).

    The structural fields (sizes, partiality, exit code) are always present.
    Phase 4 added the admission fields: ``tool_family``, ``session``,
    ``workspace``, a stable ``resource_id`` (what was read/run/searched, never
    the tool-call id), the content ``version_hash`` and the result ``content``
    / ``stderr`` themselves. :meth:`to_metadata` never includes content, so
    events and logs stay digest-only.
    """

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
    tool_family: str = "other"
    session: str = ""
    workspace: str = ""
    resource_id: str | None = None
    version_hash: str = ""
    content: str = ""
    stderr: str = ""
    source_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def call_id(self) -> str:
        return self.tool_call_id

    @property
    def truncation_reason(self) -> str | None:
        return self.partial_reason

    @property
    def is_complete(self) -> bool:
        """True only for a result known to be whole (never inferred for partials)."""
        return not self.is_partial and not self.source_truncated

    @classmethod
    def from_text(
        cls,
        *,
        tool_call_id: str,
        tool_name: str,
        text: str,
        is_error: bool = False,
        source_byte_length: int | None = None,
        stderr: str = "",
        exit_code: int | None = None,
        partial_reason: str | None = None,
        tool_family: str = "other",
        session: str = "",
        workspace: str = "",
        resource_id: str | None = None,
        source_metadata: Mapping[str, Any] | None = None,
        keep_content: bool = False,
    ) -> ToolResultEnvelope:
        """Derive the envelope deterministically from the result text.

        ``partial_reason`` / ``exit_code`` given by the caller (e.g. a host
        hook that knows the client truncated the output) win over the
        heuristics; a heuristic never clears an explicit partial flag.
        ``keep_content`` stores ``text``/``stderr`` on the envelope (the
        admission path needs them; event normalization does not).
        """
        raw = text.encode("utf-8", "replace")
        reason = partial_reason
        if reason is None:
            for marker in _TRUNCATION_MARKERS:
                if marker.search(text):
                    reason = "truncation_marker"
                    break
        source_truncated = source_byte_length is not None and source_byte_length > len(raw)
        if source_truncated and reason is None:
            reason = "source_larger_than_result"
        code = exit_code
        if code is None:
            match = _EXIT_CODE.search(text[-4000:])
            if match:
                try:
                    code = int(match.group(1))
                except ValueError:
                    code = None
        version = content_version_hash(text, stderr)
        return cls(
            v=TOOL_RESULT_ENVELOPE_VERSION,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            byte_length=len(raw),
            line_count=text.count("\n") + (1 if text else 0),
            is_error=bool(is_error) or (code is not None and code != 0),
            is_partial=reason is not None,
            partial_reason=reason,
            source_truncated=source_truncated,
            source_byte_length=source_byte_length,
            exit_code=code,
            tool_family=tool_family,
            session=session,
            workspace=workspace,
            resource_id=resource_id,
            version_hash=version,
            content=text if keep_content else "",
            stderr=stderr if keep_content else "",
            source_metadata=dict(source_metadata or {}),
        )

    def to_metadata(self) -> dict[str, Any]:
        """Structural metadata only (no content, no stderr)."""
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
        data = asdict(self)
        data["source_metadata"] = dict(self.source_metadata)
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ToolResultEnvelope:
        version = int(value.get("v", -1))
        if version not in _ENVELOPE_READABLE_VERSIONS:
            raise InterfaceVersionError(f"unsupported ToolResultEnvelope version {version!r}")
        fields = cls.__dataclass_fields__
        data = {k: value[k] for k in fields if k in value}
        data["v"] = TOOL_RESULT_ENVELOPE_VERSION
        data["source_metadata"] = dict(value.get("source_metadata") or {})
        return cls(**data)


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

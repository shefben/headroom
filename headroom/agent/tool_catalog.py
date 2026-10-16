"""Semantic progressive tool catalog (unified-21 #9).

Only where native tool search is unavailable: a request whose client already
defers (Claude Code ``ENABLE_TOOL_SEARCH``, a ``tool_search_tool_*`` /
``defer_loading`` shape) or that Headroom's own server-side deferral already
handled is left exactly as is, and a tool-free request is never given tools.

Otherwise every tool stays in the request — so each name remains callable and
``tool_choice`` / history ``tool_use`` references stay valid — but only the
resident set and the top :data:`DEFAULT_MATERIALIZE` ranked tools keep their
full (already schema-compacted) definition. The rest become compact catalog
entries: name, purpose (first sentence) and the parameter signature (required
parameters typed, optional ones listed), with their cache breakpoints kept.

Ranking: task relevance (lexical + the Rust hybrid/embedding scorer), graph
neighbours, recent successful use and a bounded learned preference; JevK5 may
break ties among the top :data:`MAX_FINALISTS` when the cut is ambiguous.

Cache rule: the materialized set is sticky per (session, client tool set).
While the prefix is cache-hot it never changes to shave nominal schema tokens;
it only grows when correctness requires it (``tool_choice`` names a tool, the
user names a tool — schema introspection — or the model called a catalog-only
tool). Output for an unchanged state is byte-identical.

Local tools use :class:`ToolRegistry` (the same interface later phases'
validator/macro tools register through); the CCR retrieval tool is registered
resident here, so the catalog never compacts it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from headroom.log_safety import describe_exception

log = logging.getLogger(__name__)

FEATURE_TOOL_CATALOG = "agent_tool_catalog"
DEFAULT_MATERIALIZE = 5
MIN_CATALOG_TOOLS = 12
MAX_FINALISTS = 16
JEV_MARGIN = 0.15
MAX_SESSIONS = 1024
CATALOG_TAG = "[catalog]"

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]{2,}")
_STOP = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "from",
        "into",
        "use",
        "when",
        "you",
        "are",
        "can",
        "will",
        "tool",
        "tools",
        "file",
        "files",
        "please",
        "need",
        "what",
        "how",
        "does",
    }
)


# ---------------------------------------------------------------------------
# Registry (local tools)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalTool:
    """A Headroom-provided tool.

    ``schema`` is the provider-neutral definition (``name``, ``description``,
    ``input_schema``); ``None`` means another component injects it (it is
    still known to the catalog, e.g. to keep it resident).
    """

    name: str
    resident: bool = True
    schema: Mapping[str, Any] | None = None
    available: Callable[[Mapping[str, Any]], bool] | None = None


#: How Claude Code names a tool served by the Headroom MCP server.
MCP_PREFIX = "mcp__headroom__"


class ToolRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tools: dict[str, LocalTool] = {}

    def register(self, tool: LocalTool) -> None:
        with self._lock:
            self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        with self._lock:
            self._tools.pop(name, None)

    def tools(self) -> list[LocalTool]:
        with self._lock:
            return [self._tools[k] for k in sorted(self._tools)]

    def get(self, name: str) -> LocalTool | None:
        """A local tool by its own name or as the host exposes it (``mcp__headroom__<name>``)."""
        with self._lock:
            tool = self._tools.get(name)
            if tool is None and name.startswith(MCP_PREFIX):
                tool = self._tools.get(name[len(MCP_PREFIX) :])
        return tool

    def is_resident(self, name: str) -> bool:
        tool = self.get(name)
        return bool(tool is not None and tool.resident)


REGISTRY = ToolRegistry()


def _register_builtin() -> None:
    from headroom.ccr.tool_injection import CCR_TOOL_NAME

    REGISTRY.register(LocalTool(CCR_TOOL_NAME, resident=True))


_register_builtin()


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def _name(tool: Mapping[str, Any], shape: str) -> str:
    if shape == "openai_chat":
        fn = tool.get("function")
        return str(fn.get("name") or "") if isinstance(fn, Mapping) else ""
    return str(tool.get("name") or "")


def _description(tool: Mapping[str, Any], shape: str) -> str:
    if shape == "openai_chat":
        fn = tool.get("function") if isinstance(tool.get("function"), Mapping) else {}
        return str(fn.get("description") or "")
    return str(tool.get("description") or "")


def _schema(tool: Mapping[str, Any], shape: str) -> Mapping[str, Any]:
    if shape == "openai_chat":
        fn = tool.get("function") if isinstance(tool.get("function"), Mapping) else {}
        value = fn.get("parameters")
    else:
        value = tool.get("input_schema")
    return value if isinstance(value, Mapping) else {}


def _shape_of(tools: Sequence[Any]) -> str | None:
    dicts = [t for t in tools if isinstance(t, Mapping)]
    if not dicts:
        return None
    if all(t.get("type") == "function" and isinstance(t.get("function"), Mapping) for t in dicts):
        return "openai_chat"
    if all(
        "function" not in t and (t.get("type") or ("name" in t and "input_schema" in t))
        for t in dicts
    ):
        return "anthropic"
    return None


def _purpose(description: str) -> str:
    text = " ".join(description.split())
    match = re.match(r"(.{8,240}?[.!?])(?:\s|$)", text)
    return (match.group(1) if match else text[:200]).strip()


def _type_of(prop: Any) -> str:
    if not isinstance(prop, Mapping):
        return "any"
    t = prop.get("type")
    if isinstance(t, list):
        return "|".join(str(x) for x in t)
    if isinstance(t, str):
        return t
    if "enum" in prop:
        return "enum"
    if "anyOf" in prop or "oneOf" in prop:
        return "union"
    return "any"


def signature(name: str, schema: Mapping[str, Any]) -> str:
    props = schema.get("properties") if isinstance(schema.get("properties"), Mapping) else {}
    required = [r for r in schema.get("required") or [] if isinstance(r, str)]
    parts = [f"{p}: {_type_of(props.get(p))}" for p in required]
    parts += [f"{p}?: {_type_of(v)}" for p, v in props.items() if p not in required]
    return f"{name}({', '.join(parts)})"


def _compact_props(schema: Mapping[str, Any]) -> dict[str, Any]:
    props = schema.get("properties") if isinstance(schema.get("properties"), Mapping) else {}
    out: dict[str, Any] = {}
    for key, prop in props.items():
        entry: dict[str, Any] = {}
        if isinstance(prop, Mapping):
            if isinstance(prop.get("type"), str | list):
                entry["type"] = prop["type"]
            enum = prop.get("enum")
            if isinstance(enum, list) and len(enum) <= 8:
                entry["enum"] = enum
        out[str(key)] = entry
    return out


def stub_tool(tool: Mapping[str, Any], shape: str) -> dict[str, Any]:
    """The compact catalog entry for ``tool`` (callable; signature preserved)."""
    name = _name(tool, shape)
    schema = _schema(tool, shape)
    compact: dict[str, Any] = {"type": "object", "properties": _compact_props(schema)}
    required = [r for r in schema.get("required") or [] if isinstance(r, str)]
    if required:
        compact["required"] = required
    if "additionalProperties" in schema:
        compact["additionalProperties"] = schema["additionalProperties"]
    desc = (
        f"{_purpose(_description(tool, shape))} {CATALOG_TAG} {signature(name, schema)}; "
        "the full schema is provided once this tool is needed."
    ).strip()
    if shape == "openai_chat":
        fn = dict(tool.get("function") or {})
        fn["description"] = desc
        fn["parameters"] = compact
        out = {k: v for k, v in tool.items() if k != "function"}
        out["function"] = fn
        return out
    out = {k: v for k, v in tool.items() if k not in ("description", "input_schema")}
    out["description"] = desc
    out["input_schema"] = compact
    return out


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text or "") if w.lower() not in _STOP}


def _name_words(name: str) -> set[str]:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", name).replace("_", " ").replace("-", " ")
    return _words(spaced)


def _hybrid(texts: list[str], query: str) -> list[float]:
    if not texts or not query.strip():
        return [0.0] * len(texts)
    try:
        from headroom import _core

        scores = list(_core.hybrid_relevance_scores([t[:2000] for t in texts], query))
    except Exception:  # noqa: BLE001 - no scorer: lexical only
        return [0.0] * len(texts)
    top = max(scores) if scores else 0.0
    return [s / top if top > 0 else 0.0 for s in scores]


def recent_tool_use(messages: Sequence[Any], shape: str) -> dict[str, float]:
    """``tool name -> recency score`` for successful calls in the history."""
    calls: list[tuple[str, str]] = []
    errors: set[str] = set()
    for msg in messages:
        if not isinstance(msg, Mapping):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for b in content:
                if not isinstance(b, Mapping):
                    continue
                if b.get("type") == "tool_use":
                    calls.append((str(b.get("id") or ""), str(b.get("name") or "")))
                elif b.get("type") == "tool_result" and b.get("is_error"):
                    errors.add(str(b.get("tool_use_id") or ""))
        for call in msg.get("tool_calls") or []:
            if isinstance(call, Mapping):
                fn = call.get("function") if isinstance(call.get("function"), Mapping) else {}
                calls.append((str(call.get("id") or ""), str(fn.get("name") or "")))
    out: dict[str, float] = {}
    recent = [c for c in calls if c[0] not in errors][-20:]
    for rank, (_cid, name) in enumerate(reversed(recent)):
        out.setdefault(name, max(0.2, 1.0 - 0.08 * rank))
    return out


def called_tools(messages: Sequence[Any]) -> list[tuple[str, str]]:
    """``(call id, tool name)`` for every tool call in the history."""
    out: list[tuple[str, str]] = []
    for msg in messages:
        if not isinstance(msg, Mapping):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for b in content:
                if isinstance(b, Mapping) and b.get("type") == "tool_use":
                    out.append((str(b.get("id") or ""), str(b.get("name") or "")))
        for call in msg.get("tool_calls") or []:
            if isinstance(call, Mapping):
                fn = call.get("function") if isinstance(call.get("function"), Mapping) else {}
                out.append((str(call.get("id") or ""), str(fn.get("name") or "")))
    return out


@dataclass
class _Session:
    digest: str
    materialized: frozenset[str]
    seen_calls: OrderedDict[str, None] = field(default_factory=OrderedDict)


@dataclass(frozen=True)
class CatalogResult:
    tools: list[Any]
    materialized: tuple[str, ...]
    stubbed: tuple[str, ...]
    reason: str
    reused: bool = False
    jev_weight: float = 0.0

    @property
    def changed(self) -> bool:
        return bool(self.stubbed)


@dataclass(frozen=True)
class CatalogRequest:
    tools: Sequence[Any]
    messages: Sequence[Any]
    session_key: str
    query: str = ""
    terms: Sequence[str] = ()
    graph_terms: Sequence[str] = ()
    tool_choice: Any = None
    prefix_hot: bool = False
    workspace: str = ""
    model: str = ""
    advisor: Any = None
    feedback: Any = None
    materialize: int = DEFAULT_MATERIALIZE


class ProgressiveToolCatalog:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, _Session] = OrderedDict()
        self.counters: dict[str, int] = {
            "applied": 0,
            "reused": 0,
            "native_search": 0,
            "tool_free": 0,
            "too_few_tools": 0,
            "unsupported_shape": 0,
            "forced_materialize": 0,
            "catalog_miss_calls": 0,
            "ranked_hits": 0,
            "stubbed_tools": 0,
        }

    # -- public ---------------------------------------------------------------

    def apply(self, req: CatalogRequest) -> CatalogResult:
        tools = list(req.tools or [])
        if not tools:
            self._bump("tool_free")
            return CatalogResult(tools, (), (), "tool_free")
        from headroom.proxy.helpers import request_already_defers_tools

        if request_already_defers_tools(tools):
            self._bump("native_search")
            return CatalogResult(tools, (), (), "native_search")
        shape = _shape_of(tools)
        if shape is None:
            self._bump("unsupported_shape")
            return CatalogResult(tools, (), (), "unsupported_shape")
        candidates = self._stubbable(tools, shape)
        if len(tools) < MIN_CATALOG_TOOLS or len(candidates) <= req.materialize:
            self._bump("too_few_tools")
            return CatalogResult(tools, (), (), "too_few_tools")
        digest = hashlib.sha256(
            json.dumps(tools, sort_keys=True, default=str).encode("utf-8", "replace")
        ).hexdigest()[:24]
        names = {_name(t, shape) for t in candidates}
        forced = self._forced(req, names)
        jev_w = 0.0
        with self._lock:
            state = self._sessions.get(req.session_key)
            if state is not None:
                self._sessions.move_to_end(req.session_key)
        self._learn_from_history(req, state, names)
        if (
            state is not None
            and state.digest == digest
            and (req.prefix_hot or not forced - state.materialized)
        ):
            materialized = state.materialized | forced
            reused = True
            if materialized != state.materialized:
                self._bump("forced_materialize")
        else:
            ranked, jev_w = self._rank(req, candidates, shape)
            chosen = [n for n, _ in ranked[: req.materialize]]
            materialized = frozenset(chosen) | forced
            if state is not None and state.digest == digest and req.prefix_hot:
                materialized |= state.materialized  # hot prefix: never shrink
            reused = False
        with self._lock:
            prev = self._sessions.get(req.session_key)
            seen = prev.seen_calls if prev is not None else OrderedDict()
            self._sessions[req.session_key] = _Session(digest, materialized, seen)
            while len(self._sessions) > MAX_SESSIONS:
                self._sessions.popitem(last=False)
        out: list[Any] = []
        stubbed: list[str] = []
        stubbable = {id(t) for t in candidates}
        for tool in tools:
            if id(tool) in stubbable and _name(tool, shape) not in materialized:
                out.append(stub_tool(tool, shape))
                stubbed.append(_name(tool, shape))
            else:
                out.append(tool)
        self._bump("reused" if reused else "applied")
        with self._lock:
            self.counters["stubbed_tools"] += len(stubbed)
        return CatalogResult(
            out,
            tuple(sorted(materialized)),
            tuple(stubbed),
            "catalog",
            reused=reused,
            jev_weight=jev_w,
        )

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {"sessions": len(self._sessions), **self.counters}

    def reset(self) -> None:
        with self._lock:
            self._sessions.clear()

    # -- internals ------------------------------------------------------------

    def _bump(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + n

    @staticmethod
    def _stubbable(tools: Sequence[Any], shape: str) -> list[Mapping[str, Any]]:
        from headroom.proxy.helpers import _tool_search_resident_key, resolved_core_tools

        core = {_tool_search_resident_key(n) for n in resolved_core_tools()}
        out: list[Mapping[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, Mapping):
                continue
            if shape == "anthropic" and tool.get("type"):
                continue  # typed server tools (web_search, computer, ...)
            name = _name(tool, shape)
            if not name or _tool_search_resident_key(name) in core or REGISTRY.is_resident(name):
                continue
            fn = tool.get("function") if shape == "openai_chat" else tool
            if isinstance(fn, Mapping) and fn.get("strict"):
                continue  # strict schemas validate exactly; never compacted
            out.append(tool)
        return out

    @staticmethod
    def _forced(req: CatalogRequest, names: set[str]) -> frozenset[str]:
        forced: set[str] = set()
        choice = req.tool_choice
        if isinstance(choice, Mapping):
            name = choice.get("name")
            if not name and isinstance(choice.get("function"), Mapping):
                name = choice["function"].get("name")
            if isinstance(name, str) and name in names:
                forced.add(name)
        # Schema introspection: a tool the user names gets its full schema.
        latest = _latest_user_text(req.messages)
        for name in names:
            if len(name) >= 3 and re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", latest):
                forced.add(name)
        return frozenset(forced)

    def _learn_from_history(
        self, req: CatalogRequest, state: _Session | None, names: set[str]
    ) -> None:
        """Calls since the last request: catalog misses vs ranked hits."""
        if state is None:
            return
        from headroom.agent.feedback import FeedbackKey, opaque, workspace_hash

        new: list[tuple[str, str]] = []
        with self._lock:
            for cid, name in called_tools(req.messages):
                if cid and cid not in state.seen_calls:
                    state.seen_calls[cid] = None
                    new.append((cid, name))
            while len(state.seen_calls) > 512:
                state.seen_calls.popitem(last=False)
        ws = workspace_hash(req.workspace)
        materialized: set[str] = set(state.materialized)
        for _cid, name in new:
            if name not in names:
                continue
            hit = name in materialized
            self._bump("ranked_hits" if hit else "catalog_miss_calls")
            if req.feedback is not None:
                req.feedback.observe(
                    FeedbackKey("catalog", "tool", "", opaque(name), req.model, "", ws),
                    1.0 if hit else 0.0,
                )
            if hit:
                from headroom.agent.arbiter import get_arbiter

                # A ranked (non-core) schema was there when needed: one
                # discovery round-trip estimated avoided.
                get_arbiter().note_avoided_turns(1.0)
            else:
                # The model needed a catalog-only tool: materialize it now.
                materialized.add(name)
        if materialized != set(state.materialized):
            with self._lock:
                state.materialized = frozenset(materialized)
            self._bump("forced_materialize")

    def _rank(
        self, req: CatalogRequest, candidates: list[Mapping[str, Any]], shape: str
    ) -> tuple[list[tuple[str, float]], float]:
        from headroom.agent.feedback import FeedbackKey, opaque, workspace_hash

        query = req.query or " ".join(req.terms)
        qwords = _words(query) | {t.lower() for t in req.terms if len(t) >= 3}
        gwords = {g.lower() for g in req.graph_terms}
        texts = [f"{_name(t, shape)}: {_description(t, shape)[:600]}" for t in candidates]
        hybrid = _hybrid(texts, query)
        recent = recent_tool_use(req.messages, shape)
        ws = workspace_hash(req.workspace)
        scored: list[tuple[str, float]] = []
        for i, tool in enumerate(candidates):
            name = _name(tool, shape)
            words = _name_words(name) | _words(_description(tool, shape)[:600])
            lexical = len(qwords & words) / max(1, min(len(qwords), 8)) if qwords else 0.0
            graph = len(gwords & words) / max(1, len(gwords)) if gwords else 0.0
            prior = 0.5
            if req.feedback is not None:
                prior = req.feedback.prior(
                    FeedbackKey("catalog", "tool", "", opaque(name), req.model, "", ws)
                )
            score = (
                0.40 * min(1.0, lexical)
                + 0.25 * hybrid[i]
                + 0.10 * min(1.0, graph)
                + 0.25 * recent.get(name, 0.0)
                + 0.20 * (prior - 0.5)
            )
            scored.append((name, score))
        scored.sort(key=lambda x: (-round(x[1], 9), x[0]))
        k = req.materialize
        jev_w = 0.0
        if (
            req.advisor is not None
            and len(scored) > k
            and (not qwords or scored[k - 1][1] - scored[k][1] < JEV_MARGIN)
        ):
            jev_w, scored = self._jev_ties(req, scored, candidates, shape)
        return scored, jev_w

    @staticmethod
    def _jev_ties(
        req: CatalogRequest,
        scored: list[tuple[str, float]],
        candidates: list[Mapping[str, Any]],
        shape: str,
    ) -> tuple[float, list[tuple[str, float]]]:
        finalists = scored[:MAX_FINALISTS]
        purposes = {_name(t, shape): _purpose(_description(t, shape))[:120] for t in candidates}
        options = {f"t{i}": f"{n}: {purposes.get(n, '')}" for i, (n, _) in enumerate(finalists)}
        try:
            budget = req.advisor.request_budget()
            advice = req.advisor.choose(
                "tool_catalog",
                {"task": (req.query or " ".join(req.terms))[:600], "tools": options},
                "Which tool is the agent most likely to need next for this task?",
                options,
                budget=budget,
            )
        except Exception as exc:  # noqa: BLE001 - advisory only
            log.debug("catalog advisor failed: %s", describe_exception(exc))
            return 0.0, scored
        if advice is None:
            return 0.0, scored
        w = max(0.0, min(0.40, float(getattr(advice, "weight", 0.0) or 0.0)))
        probs = getattr(advice, "probabilities", {}) or {}
        blended = [
            (n, (1 - w) * s + w * float(probs.get(f"t{i}", 0.0) or 0.0))
            for i, (n, s) in enumerate(finalists)
        ]
        blended.sort(key=lambda x: (-round(x[1], 9), x[0]))
        return w, blended + scored[MAX_FINALISTS:]


def _latest_user_text(messages: Sequence[Any]) -> str:
    for msg in reversed(messages):
        if not isinstance(msg, Mapping) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(b.get("text") or "")
                for b in content
                if isinstance(b, Mapping) and b.get("type") == "text"
            )
    return ""


_CATALOG: ProgressiveToolCatalog | None = None
_CATALOG_LOCK = threading.Lock()


def get_catalog() -> ProgressiveToolCatalog:
    global _CATALOG
    if _CATALOG is None:
        with _CATALOG_LOCK:
            if _CATALOG is None:
                _CATALOG = ProgressiveToolCatalog()
    return _CATALOG


def apply_request_catalog(
    *,
    config: Any,
    tools: Any,
    messages: Sequence[Any],
    session_key: str | None,
    tool_choice: Any = None,
    prefix_hot: bool = False,
    model: str = "",
    request_id: str | None = None,
) -> CatalogResult | None:
    """Handler entry point; ``None`` when the feature is off. Never raises."""
    rollout = getattr(config, "rollout", None)
    try:
        if rollout is None or not rollout.is_enabled(FEATURE_TOOL_CATALOG):
            return None
    except KeyError:
        return None
    if not isinstance(tools, list) or not tools:
        get_catalog()._bump("tool_free")
        return None
    try:
        from headroom.agent.hook import _active_runtime, latest_session_query
        from headroom.agent.identity import resolve_identity
        from headroom.proxy.project_context import (
            get_current_project,
            get_registered_cwd,
            get_registered_session,
        )

        identity = resolve_identity(
            session_token=get_registered_session(),
            registered_cwd=get_registered_cwd(),
            attribution=get_current_project(),
        )
        relevance = latest_session_query(identity.session_id)
        query = relevance.text if relevance is not None else _latest_user_text(messages)
        terms = tuple(relevance.terms) if relevance is not None else ()
        graph = tuple(relevance.graph_neighbors) if relevance is not None else ()
        runtime = _active_runtime()
        advisor = getattr(runtime, "advisor", None) if runtime is not None else None
        feedback = None
        try:
            if rollout.is_enabled("agent_feedback"):
                from headroom.agent.feedback import get_feedback

                feedback = get_feedback()
        except KeyError:
            feedback = None
        return get_catalog().apply(
            CatalogRequest(
                tools=tools,
                messages=messages,
                session_key=f"{identity.workspace_id}|{session_key or identity.session_id}",
                query=query,
                terms=terms,
                graph_terms=graph,
                tool_choice=tool_choice,
                prefix_hot=prefix_hot,
                workspace=identity.workspace_id,
                model=model,
                advisor=advisor,
                feedback=feedback,
            )
        )
    except Exception as exc:  # noqa: BLE001 - the catalog never fails a request
        log.warning("[%s] tool catalog failed: %s", request_id, describe_exception(exc))
        return None


__all__ = [
    "CatalogRequest",
    "CatalogResult",
    "DEFAULT_MATERIALIZE",
    "FEATURE_TOOL_CATALOG",
    "LocalTool",
    "ProgressiveToolCatalog",
    "REGISTRY",
    "ToolRegistry",
    "apply_request_catalog",
    "get_catalog",
    "signature",
    "stub_tool",
]

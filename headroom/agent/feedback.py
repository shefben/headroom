"""Retrieval-feedback / retention adaptation (unified-21 #12).

Every lossy admission the arbiter makes registers one *pending observation*
keyed only by a generalized :class:`FeedbackKey` ``(tool family, content type,
task class, strategy, model, language)`` plus an optional workspace hash —
never raw text, paths or tool output. An observation resolves exactly once:

* negative (0.0) on CCR span re-retrieval of its hash, an immediate reread /
  re-search of the same resource, a failed repair (edit error) on a path whose
  read was shown lossy, or a user correction while it is pending;
* 0.25 on a failing test run while it is pending (a false candidate);
* positive (1.0) when :data:`SETTLE_REQUESTS` later requests in the same scope
  pass with no negative signal.

Priors move by ``prior <- clamp(.95*prior + .05*outcome, .1, .9)`` and only
once a key has at least :data:`MIN_OBSERVATIONS` independent observations
(the first ones are buffered, then applied in order). A prior can tweak
arbiter / catalog preference within hard policy (its weight is .07 in the
arbiter); it never overrides an invariant or explicitly required detail.

Reversible: :meth:`RetentionFeedback.revert` restores earlier priors from a
bounded history, :meth:`reset` forgets a key (or everything), and disabling
the ``agent_feedback`` rollout feature makes every prior the neutral 0.5.
Persistence (optional) goes through the agent ledger on the runtime worker.

Holdout and automatic family revert (Phase 8). ``HEADROOM_TRANSFORM_HOLDOUT``
(a fraction, default 0, at most .5, read live) holds a scope-stable share of
sessions out of lossy admission: they get the exact original, and the lossy
form the arbiter *would* have used registers a ``control`` observation that
resolves on the same signals. Per tool family, once the treatment arm has
:data:`REVERT_MIN_OBSERVATIONS` resolved observations and its negative rate
exceeds the control arm's by :data:`REVERT_MARGIN` (or, without enough control
observations, reaches :data:`REVERT_ABSOLUTE`), the family is *reverted*: the
arbiter still evaluates it but admits only reversible forms, so the feature
stays available and safe. Control observations never move the strategy
priors. :meth:`RetentionFeedback.reinstate` (or :meth:`reset`) clears a
revert; reverts are process-local and a restart re-measures.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from headroom.log_safety import describe_exception

log = logging.getLogger(__name__)

FEATURE_FEEDBACK = "agent_feedback"

NEUTRAL_PRIOR = 0.5
PRIOR_MIN = 0.1
PRIOR_MAX = 0.9
DECAY = 0.95
LEARNING = 0.05
MIN_OBSERVATIONS = 5
SETTLE_REQUESTS = 3
HISTORY = 32
MAX_KEYS = 4096
MAX_PENDING = 8192

OUTCOME_NEGATIVE = 0.0
OUTCOME_FALSE_CANDIDATE = 0.25
OUTCOME_POSITIVE = 1.0

ARM_TREATMENT = "treatment"
ARM_CONTROL = "control"
HOLDOUT_ENV = "HEADROOM_TRANSFORM_HOLDOUT"
HOLDOUT_MAX = 0.5
REVERT_MIN_OBSERVATIONS = 8
REVERT_MARGIN = 0.25
REVERT_ABSOLUTE = 0.5


def transform_holdout_fraction() -> float:
    """``HEADROOM_TRANSFORM_HOLDOUT`` clamped to [0, .5]; unset/invalid is 0."""
    from headroom.proxy import runtime_env

    try:
        value = float(runtime_env.getenv(HOLDOUT_ENV, "0") or "0")
    except ValueError:
        return 0.0
    if value != value:  # NaN
        return 0.0
    return max(0.0, min(HOLDOUT_MAX, value))


def transform_arm(scope: str) -> str:
    """Scope-stable treatment/control assignment for lossy tool admission."""
    from headroom.proxy.output_savings_policy import assign_arm

    return assign_arm("transform:" + (scope or ""), transform_holdout_fraction())


_TABLE = (
    "CREATE TABLE IF NOT EXISTS agent_feedback_priors ("
    " key TEXT PRIMARY KEY, prior REAL NOT NULL, observations INTEGER NOT NULL,"
    " buffered TEXT NOT NULL, updated_at REAL NOT NULL)"
)


def workspace_hash(workspace_id: str | None) -> str:
    if not workspace_id:
        return ""
    return hashlib.sha256(workspace_id.encode("utf-8", "replace")).hexdigest()[:12]


def opaque(name: str) -> str:
    """A non-reversible tag for a value that may be private (tool names)."""
    return hashlib.sha256((name or "").encode("utf-8", "replace")).hexdigest()[:10]


@dataclass(frozen=True)
class FeedbackKey:
    family: str
    content_type: str = ""
    task_class: str = ""
    strategy: str = ""
    model: str = ""
    language: str = ""
    workspace: str = ""  # workspace_hash(), or "" for the global prior

    def ident(self) -> str:
        return "|".join(
            (
                self.family,
                self.content_type,
                self.task_class,
                self.strategy,
                _model_class(self.model),
                self.language,
                self.workspace,
            )
        )

    def global_key(self) -> FeedbackKey:
        return FeedbackKey(
            self.family,
            self.content_type,
            self.task_class,
            self.strategy,
            self.model,
            self.language,
            "",
        )


def _model_class(model: str) -> str:
    """Generalize a model id to its family (no deployment-specific ids)."""
    name = (model or "").lower()
    for fam in ("opus", "sonnet", "haiku", "gpt-5", "gpt-4", "o3", "o4", "gemini", "codex"):
        if fam in name:
            return fam
    return "other" if name else ""


@dataclass
class _Prior:
    prior: float = NEUTRAL_PRIOR
    observations: int = 0
    buffered: list[float] = field(default_factory=list)
    history: deque[float] = field(default_factory=lambda: deque(maxlen=HISTORY))


@dataclass
class _Pending:
    key: FeedbackKey
    scope: str
    seq: int
    ccr_hash: str = ""
    resource_id: str = ""
    path: str = ""
    arm: str = ARM_TREATMENT


def _clamp(x: float) -> float:
    return max(PRIOR_MIN, min(PRIOR_MAX, x))


class RetentionFeedback:
    """Process-wide, thread-safe feedback store (see module docstring)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._priors: OrderedDict[str, _Prior] = OrderedDict()
        self._pending: OrderedDict[str, _Pending] = OrderedDict()
        self._by_hash: dict[str, str] = {}
        self._by_resource: dict[tuple[str, str], str] = {}
        self._by_path: dict[tuple[str, str], str] = {}
        self._seq: dict[str, int] = {}
        self._last_user: dict[str, str] = {}
        self.enabled = True
        self._db: Any = None
        self._submit: Callable[[Callable[[], Any]], bool] | None = None
        self.counters: dict[str, int] = {
            "registered": 0,
            "resolved_positive": 0,
            "resolved_negative": 0,
            "resolved_false_candidate": 0,
            "retrievals": 0,
            "rereads": 0,
            "repairs": 0,
            "corrections": 0,
            "updates": 0,
            "reverts": 0,
            "control_registered": 0,
            "family_reverts": 0,
        }
        #: (tool family, arm) -> [resolved, negative]
        self._arms: dict[tuple[str, str], list[int]] = {}
        self._reverted: dict[str, dict[str, Any]] = {}

    # -- persistence ------------------------------------------------------------

    def bind(self, ledger: Any, submit: Callable[[Callable[[], Any]], bool]) -> None:
        with ledger.lock:
            ledger.connection.execute(_TABLE)
        with self._lock:
            self._db, self._submit = ledger, submit
        submit(lambda: self._load(ledger))

    def unbind(self) -> None:
        with self._lock:
            self._db, self._submit = None, None

    def _load(self, db: Any) -> None:
        try:
            rows = db.query(
                "SELECT key, prior, observations, buffered FROM agent_feedback_priors"
                " ORDER BY updated_at DESC LIMIT ?",
                (MAX_KEYS,),
            )
        except Exception as exc:  # noqa: BLE001 - memory-only priors still work
            log.warning("feedback priors load failed: %s", describe_exception(exc))
            return
        with self._lock:
            for key, prior, obs, buffered in rows:
                if key in self._priors:
                    continue
                try:
                    buf = [float(v) for v in json.loads(buffered or "[]")][:MIN_OBSERVATIONS]
                except (TypeError, ValueError):
                    buf = []
                self._priors[str(key)] = _Prior(_clamp(float(prior)), int(obs), buf)

    def _persist(self, ident: str, state: _Prior) -> None:
        with self._lock:
            submit, db = self._submit, self._db
            row = (ident, state.prior, state.observations, json.dumps(state.buffered), time.time())
        if submit is None or db is None:
            return

        def write() -> None:
            try:
                db.execute(
                    "INSERT OR REPLACE INTO agent_feedback_priors(key, prior, observations,"
                    " buffered, updated_at) VALUES (?,?,?,?,?)",
                    row,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("feedback prior persist failed: %s", describe_exception(exc))

        submit(write)

    # -- reads --------------------------------------------------------------------

    def prior(self, key: FeedbackKey) -> float:
        """Workspace prior when it has enough evidence, else the global one."""
        if not self.enabled:
            return NEUTRAL_PRIOR
        with self._lock:
            for k in (key, key.global_key()) if key.workspace else (key,):
                state = self._priors.get(k.ident())
                if state is not None and state.observations >= MIN_OBSERVATIONS:
                    return state.prior
        return NEUTRAL_PRIOR

    def observations(self, key: FeedbackKey) -> int:
        with self._lock:
            state = self._priors.get(key.ident())
            return state.observations if state else 0

    def retrieval_rate(self, key: FeedbackKey) -> float | None:
        """Estimated re-retrieval probability (1 - prior) once learned, else None."""
        if not self.enabled:
            return None
        with self._lock:
            state = self._priors.get(key.ident()) or self._priors.get(key.global_key().ident())
            if state is None or state.observations < MIN_OBSERVATIONS:
                return None
            return 1.0 - state.prior

    # -- registration and signals ---------------------------------------------------

    def register(
        self,
        scope: str,
        key: FeedbackKey,
        *,
        ccr_hash: str = "",
        resource_id: str = "",
        path: str = "",
        arm: str = ARM_TREATMENT,
    ) -> None:
        """One lossy decision (or, for ``arm="control"``, a held-out one) awaits its outcome."""
        if not self.enabled:
            return
        arm = ARM_CONTROL if arm == ARM_CONTROL else ARM_TREATMENT
        with self._lock:
            seq = self._seq.get(scope, 0)
            obs_id = f"{scope}|{arm}|{ccr_hash or resource_id}|{seq}"
            # A newer decision on the same resource supersedes the old pending one.
            if resource_id:
                old = self._by_resource.get((scope, resource_id))
                if old is not None:
                    self._drop(old)
            self._pending[obs_id] = _Pending(key, scope, seq, ccr_hash, resource_id, path, arm)
            if arm == ARM_CONTROL:
                self.counters["control_registered"] += 1
            if ccr_hash:
                self._by_hash[ccr_hash] = obs_id
            if resource_id:
                self._by_resource[(scope, resource_id)] = obs_id
            if path:
                self._by_path[(scope, path)] = obs_id
            while len(self._pending) > MAX_PENDING:
                self._drop(next(iter(self._pending)))
            self.counters["registered"] += 1

    def tick(self, scope: str) -> None:
        """One request in ``scope``; settles quiet observations as positive."""
        if not self.enabled:
            return
        with self._lock:
            seq = self._seq.get(scope, 0) + 1
            self._seq[scope] = seq
            due = [
                oid
                for oid, p in self._pending.items()
                if p.scope == scope and seq - p.seq > SETTLE_REQUESTS
            ]
        for oid in due:
            self._resolve(oid, OUTCOME_POSITIVE)

    def signal_retrieval(self, ccr_hash: str) -> bool:
        with self._lock:
            oid = self._by_hash.get(ccr_hash)
        if oid is None:
            return False
        self.counters["retrievals"] += 1
        return self._resolve(oid, OUTCOME_NEGATIVE)

    def signal_reread(self, scope: str, resource_id: str) -> bool:
        with self._lock:
            oid = self._by_resource.get((scope, resource_id))
        if oid is None:
            return False
        self.counters["rereads"] += 1
        return self._resolve(oid, OUTCOME_NEGATIVE)

    def signal_repair(self, scope: str, path: str) -> bool:
        with self._lock:
            oid = self._by_path.get((scope, path))
        if oid is None:
            return False
        self.counters["repairs"] += 1
        return self._resolve(oid, OUTCOME_NEGATIVE)

    def signal_test_failure(self, scope: str) -> int:
        return self._resolve_scope(scope, OUTCOME_FALSE_CANDIDATE)

    def signal_correction(self, scope: str, user_digest: str) -> int:
        """A user correction; counted once per distinct user message."""
        with self._lock:
            if self._last_user.get(scope) == user_digest:
                return 0
            self._last_user[scope] = user_digest
            while len(self._last_user) > MAX_PENDING:
                self._last_user.pop(next(iter(self._last_user)))
        self.counters["corrections"] += 1
        return self._resolve_scope(scope, OUTCOME_NEGATIVE)

    def _resolve_scope(self, scope: str, outcome: float) -> int:
        with self._lock:
            ids = [oid for oid, p in self._pending.items() if p.scope == scope]
        return sum(1 for oid in ids if self._resolve(oid, outcome))

    # -- updates ------------------------------------------------------------------

    def _drop(self, oid: str) -> _Pending | None:
        p = self._pending.pop(oid, None)
        if p is None:
            return None
        if p.ccr_hash and self._by_hash.get(p.ccr_hash) == oid:
            del self._by_hash[p.ccr_hash]
        if p.resource_id and self._by_resource.get((p.scope, p.resource_id)) == oid:
            del self._by_resource[(p.scope, p.resource_id)]
        if p.path and self._by_path.get((p.scope, p.path)) == oid:
            del self._by_path[(p.scope, p.path)]
        return p

    def _resolve(self, oid: str, outcome: float) -> bool:
        with self._lock:
            p = self._drop(oid)
        if p is None:
            return False
        name = {
            OUTCOME_POSITIVE: "resolved_positive",
            OUTCOME_FALSE_CANDIDATE: "resolved_false_candidate",
        }.get(outcome, "resolved_negative")
        self.counters[name] += 1
        if p.arm != ARM_CONTROL:
            # Control observations measure the baseline only; the strategy was not shown.
            keys: Iterable[FeedbackKey] = (
                (p.key, p.key.global_key()) if p.key.workspace else (p.key,)
            )
            for key in keys:
                self.observe(key, outcome)
        self._note_arm(p.key.family, p.arm, outcome < OUTCOME_POSITIVE)
        return True

    # -- holdout comparison and family revert -------------------------------------

    def _note_arm(self, family: str, arm: str, negative: bool) -> None:
        reverted: dict[str, Any] | None = None
        with self._lock:
            stats = self._arms.setdefault((family, arm), [0, 0])
            stats[0] += 1
            stats[1] += int(negative)
            if family in self._reverted:
                return
            treat = self._arms.get((family, ARM_TREATMENT))
            if not treat or treat[0] < REVERT_MIN_OBSERVATIONS:
                return
            treat_rate = treat[1] / treat[0]
            control = self._arms.get((family, ARM_CONTROL))
            control_rate = (
                control[1] / control[0]
                if control and control[0] >= REVERT_MIN_OBSERVATIONS
                else None
            )
            if control_rate is not None:
                regressing = treat_rate >= control_rate + REVERT_MARGIN
            else:
                regressing = treat_rate >= REVERT_ABSOLUTE
            if not regressing:
                return
            reverted = {
                "treatment_negative_rate": round(treat_rate, 4),
                "control_negative_rate": None if control_rate is None else round(control_rate, 4),
                "treatment_resolved": treat[0],
                "control_resolved": control[0] if control else 0,
                "at": time.time(),
            }
            self._reverted[family] = reverted
            self.counters["family_reverts"] += 1
        control_text = reverted["control_negative_rate"]
        log.info(
            "tool family %s reverted to reversible-only admission (negative rate %.2f vs %s)",
            family,
            reverted["treatment_negative_rate"],
            "n/a" if control_text is None else control_text,
        )

    def family_reverted(self, family: str) -> bool:
        """True when ``family`` regressed: only reversible forms may be admitted."""
        if not self.enabled or not family:
            return False
        with self._lock:
            return family in self._reverted

    def reinstate(self, family: str | None = None) -> None:
        """Clear a family revert (or all) and restart its arm comparison."""
        with self._lock:
            families = [family] if family is not None else list(self._reverted)
            for fam in families:
                self._reverted.pop(fam, None)
                self._arms.pop((fam, ARM_TREATMENT), None)
                self._arms.pop((fam, ARM_CONTROL), None)

    def observe(self, key: FeedbackKey, outcome: float) -> float:
        """Record one independent observation; returns the (possibly unchanged) prior."""
        outcome = max(0.0, min(1.0, float(outcome)))
        ident = key.ident()
        with self._lock:
            state = self._priors.get(ident)
            if state is None:
                state = _Prior()
                self._priors[ident] = state
                while len(self._priors) > MAX_KEYS:
                    self._priors.popitem(last=False)
            self._priors.move_to_end(ident)
            state.observations += 1
            if state.observations < MIN_OBSERVATIONS:
                state.buffered.append(outcome)
            else:
                pending = [*state.buffered, outcome]
                state.buffered = []
                for value in pending:
                    state.history.append(state.prior)
                    state.prior = _clamp(DECAY * state.prior + LEARNING * value)
                self.counters["updates"] += 1
            prior = state.prior
        self._persist(ident, state)
        return prior

    def revert(self, key: FeedbackKey, steps: int = 1) -> float:
        with self._lock:
            state = self._priors.get(key.ident())
            if state is None:
                return NEUTRAL_PRIOR
            for _ in range(max(0, steps)):
                if not state.history:
                    break
                state.prior = state.history.pop()
            self.counters["reverts"] += 1
            prior = state.prior
        self._persist(key.ident(), state)
        return prior

    def reset(self, key: FeedbackKey | None = None) -> None:
        """Forget one key's prior (or everything, including pending observations)."""
        with self._lock:
            if key is None:
                self._priors.clear()
                for oid in list(self._pending):
                    self._drop(oid)
                self._seq.clear()
                self._last_user.clear()
                self._arms.clear()
                self._reverted.clear()
            else:
                self._priors.pop(key.ident(), None)
            db, submit = self._db, self._submit
        if db is None or submit is None:
            return
        if key is None:
            submit(lambda: db.execute("DELETE FROM agent_feedback_priors"))
        else:
            ident = key.ident()
            submit(lambda: db.execute("DELETE FROM agent_feedback_priors WHERE key=?", (ident,)))

    def status(self) -> dict[str, Any]:
        with self._lock:
            learned = sum(1 for s in self._priors.values() if s.observations >= MIN_OBSERVATIONS)
            total_resolved = (
                self.counters["resolved_positive"]
                + self.counters["resolved_negative"]
                + self.counters["resolved_false_candidate"]
            )
            return {
                "enabled": self.enabled,
                "keys": len(self._priors),
                "learned_keys": learned,
                "pending": len(self._pending),
                "persistent": self._db is not None,
                "holdout_fraction": transform_holdout_fraction(),
                "arms": {
                    f"{fam}:{arm}": {"resolved": s[0], "negative": s[1]}
                    for (fam, arm), s in sorted(self._arms.items())
                },
                "reverted_families": {k: dict(v) for k, v in sorted(self._reverted.items())},
                "re_retrieval_rate": round(self.counters["retrievals"] / total_resolved, 4)
                if total_resolved
                else 0.0,
                **self.counters,
            }


_FEEDBACK: RetentionFeedback | None = None
_FEEDBACK_LOCK = threading.Lock()


def get_feedback() -> RetentionFeedback:
    global _FEEDBACK
    if _FEEDBACK is None:
        with _FEEDBACK_LOCK:
            if _FEEDBACK is None:
                _FEEDBACK = RetentionFeedback()
    return _FEEDBACK


def reset_feedback() -> None:
    global _FEEDBACK
    with _FEEDBACK_LOCK:
        _FEEDBACK = None


def note_ccr_retrieval(ccr_hash: str) -> None:
    """CCR retrieval hook (full or selective); never raises."""
    fb = _FEEDBACK
    if fb is None or not ccr_hash:
        return
    try:
        fb.signal_retrieval(str(ccr_hash))
    except Exception as exc:  # noqa: BLE001
        log.debug("feedback retrieval signal failed: %s", describe_exception(exc))


__all__ = [
    "ARM_CONTROL",
    "ARM_TREATMENT",
    "FEATURE_FEEDBACK",
    "FeedbackKey",
    "MIN_OBSERVATIONS",
    "RetentionFeedback",
    "get_feedback",
    "note_ccr_retrieval",
    "opaque",
    "reset_feedback",
    "transform_arm",
    "transform_holdout_fraction",
    "workspace_hash",
]

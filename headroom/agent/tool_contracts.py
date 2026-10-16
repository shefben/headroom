"""Tool Contract and Argument Validator (unified-21 #19).

Every tool call maps to one canonical :class:`ToolContract` by *family*
(Read, Edit/Write, Bash/exec, Grep/Glob, Git, test/build, HTTP/API) from the
Claude Code / Codex / OpenCode tool names, without rewriting any provider
schema. Unknown tools are observed and never certified.

:meth:`ToolContractValidator.evaluate` returns a :class:`Decision`
``ALLOW | WARN | BLOCK | REPAIR_SUGGEST`` with a stable ``reason_code``, the
validated arguments and the task / scope revisions it was computed against.
Checks (deterministic, bounded, no execution):

* tool-schema syntax: required arguments, JSON types, enums, mutually
  exclusive arguments (the client's own ``input_schema`` when the proxy has
  it, else the host's documented required arguments);
* cwd existence (including ``cd X &&`` preludes), input-file existence for
  reads/edits, workspace containment, symlink escape and UNC targets;
* command availability, shell built-ins included, and platform-specific
  switches (cmd.exe ``/X`` switches in a POSIX shell, ``rm -rf`` in
  PowerShell);
* Git ref-name syntax (``git check-ref-format`` rules), with the change-scope
  firewall (#23) deciding destructive Git state changes;
* opportunistic privilege escalation (``sudo``/``runas``/``-Verb RunAs``
  the user never asked for);
* malformed test invocations (missing test paths, ``-k`` without an
  expression, no manifest for ``npm test`` / ``cargo test``).

Mutating calls are authorized by the :class:`ChangeScopeFirewall`; the
contract decision carries its revisions. External paths the user named are
never prohibited by assumption; only a trusted workspace can be protected.

Repairs are *suggestions only*: a typo'd path or command yields
``REPAIR_SUGGEST`` with the safe alternative in the diagnostic, and the call
itself is never transformed or re-run. Learned recoveries reuse
:class:`~headroom.memory.traffic_learner.TrafficLearner`'s error→recovery
pairing and :func:`~headroom.learn._shared.classify_error`; a recovery is
promoted only after three corroborating successful recoveries in the same
environment fingerprint, expires after a TTL, and is invalidated by
contradicting evidence. JevK5 may only rank already-safe suggestions.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from headroom.agent.change_scope import (
    ScopeDecision,
    Verdict,
    canonicalize,
    command_segments,
    describe,
    shell_cwd,
    shell_of,
    split_command,
)
from headroom.agent.events import EventEnvelope, EventKind, Origin, make_event
from headroom.log_safety import describe_exception

log = logging.getLogger(__name__)

FEATURE_TOOL_CONTRACTS = "agent_tool_contracts"
SCHEMA_VERSION = 1
PROMOTION_THRESHOLD = 3
RECOVERY_TTL_S = 7 * 24 * 3600.0
_HISTORY_LIMIT = 20
_RECOVERIES_LIMIT = 2048
_SESSIONS_LIMIT = 512
_TYPO_DIR_LIMIT = 2000
MAX_MESSAGE_CHARS = 560


class Family(str, Enum):
    READ = "read"
    EDIT = "edit"
    EXEC = "exec"
    SEARCH = "search"
    GIT = "git"
    TEST = "test"
    HTTP = "http"
    UNKNOWN = "unknown"


class DecisionKind(str, Enum):
    ALLOW = "ALLOW"
    WARN = "WARN"
    BLOCK = "BLOCK"
    REPAIR_SUGGEST = "REPAIR_SUGGEST"


@dataclass(frozen=True)
class ToolContract:
    tool_family: Family
    source: str
    operation_class: str  # read | mutate | execute | query | network
    side_effect: str  # none | filesystem | process | vcs | network
    schema_version: int
    hard_rules: tuple[str, ...]
    advisory_rules: tuple[str, ...]
    learned_recovery: tuple[str, ...] = ()
    tool_name: str = ""
    certified: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_family": self.tool_family.value,
            "source": self.source,
            "operation_class": self.operation_class,
            "side_effect": self.side_effect,
            "schema_version": self.schema_version,
            "hard_rules": list(self.hard_rules),
            "advisory_rules": list(self.advisory_rules),
            "learned_recovery": list(self.learned_recovery),
            "tool_name": self.tool_name,
            "certified": self.certified,
        }


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str  # hard | advisory
    message: str
    repair: str | None = None


@dataclass(frozen=True)
class Decision:
    kind: DecisionKind
    reason_code: str
    validated_arguments: Mapping[str, Any]
    task_revision: int
    scope_revision: int
    contract: ToolContract
    findings: tuple[Finding, ...] = ()
    repairs: tuple[str, ...] = ()
    scope: tuple[ScopeDecision, ...] = ()
    enforcement: str = "observe_only"

    def message(self) -> str:
        """Compact, reasoned diagnostic (empty for ALLOW)."""
        if self.kind is DecisionKind.ALLOW:
            return ""
        parts: list[str] = []
        for scope in self.scope:
            if (
                scope.verdict is not Verdict.ALLOW
                and scope.reason_code != "scope.untrusted_workspace"
            ):
                parts.append(f"{describe(scope)} [{scope.reason_code}]")
        for finding in self.findings:
            parts.append(f"{finding.message} [{finding.code}]")
        if self.repairs:
            options = " | ".join(f"`{r}`" for r in self.repairs[:3])
            parts.append(f"suggested (not run): {options}")
        text = "; ".join(dict.fromkeys(parts))
        return text if len(text) <= MAX_MESSAGE_CHARS else text[: MAX_MESSAGE_CHARS - 1] + "…"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "reason_code": self.reason_code,
            "task_revision": self.task_revision,
            "scope_revision": self.scope_revision,
            "contract": self.contract.to_dict(),
            "findings": [f.code for f in self.findings],
            "repairs": list(self.repairs),
            "scope": [s.to_dict() for s in self.scope],
            "enforcement": self.enforcement,
        }


# ---------------------------------------------------------------------------
# Family mapping (Claude Code / Codex / OpenCode names; schemas untouched)
# ---------------------------------------------------------------------------

_READ = {
    "read",
    "read_file",
    "view",
    "view_file",
    "readfile",
    "cat",
    "read_many_files",
    "notebookread",
}
_EDIT = {
    "edit",
    "write",
    "multiedit",
    "notebookedit",
    "apply_patch",
    "str_replace_based_edit_tool",
    "str_replace_editor",
    "write_file",
    "edit_file",
    "create_file",
    "replace_in_file",
    "patch",
}
_EXEC = {
    "bash",
    "powershell",
    "shell",
    "local_shell",
    "exec_command",
    "run_terminal_cmd",
    "terminal",
    "run_command",
    "shell_command",
    "run_shell_command",
    "execute_command",
}
_SEARCH = {
    "grep",
    "glob",
    "search",
    "rg",
    "list",
    "ls",
    "list_directory",
    "list_dir",
    "find_files",
    "search_files",
    "codebase_search",
    "file_search",
}
_HTTP = {"webfetch", "web_fetch", "fetch", "websearch", "web_search", "http_request"}

_TEST_CMD = re.compile(
    r"(?:^|[\s;&|(])(?:python\d?(?:\.\d+)?\s+-m\s+)?(pytest|py\.test|cargo\s+(?:test|nextest)|"
    r"npm\s+(?:run\s+)?test|yarn\s+test|pnpm\s+(?:run\s+)?test|go\s+test|jest|vitest|"
    r"mvn\s+test|gradle\s+test|ctest|tox|nox|dotnet\s+test|cargo\s+build|npm\s+run\s+build|make)\b"
)


def family_of(tool_name: str, args: Mapping[str, Any]) -> Family:
    lname = (tool_name or "").lower()
    if lname in _READ:
        return Family.READ
    if lname in _EDIT:
        return Family.EDIT
    if lname in _SEARCH:
        return Family.SEARCH
    if lname in _HTTP:
        return Family.HTTP
    if lname in _EXEC:
        command = _command(args)
        head = _first_word(command)
        if head == "git":
            return Family.GIT
        if command and _TEST_CMD.search(command):
            return Family.TEST
        return Family.EXEC
    return Family.UNKNOWN


_CONTRACTS: dict[Family, tuple[str, str, tuple[str, ...], tuple[str, ...]]] = {
    Family.READ: ("read", "none", ("required_args", "arg_types", "input_exists"), ("workspace",)),
    Family.EDIT: (
        "mutate",
        "filesystem",
        ("required_args", "arg_types", "input_exists", "change_scope"),
        ("distinct_edit", "unc_target"),
    ),
    Family.EXEC: (
        "execute",
        "process",
        ("required_args", "cwd_exists", "change_scope", "privilege"),
        ("command_available", "platform_switch"),
    ),
    Family.GIT: (
        "execute",
        "vcs",
        ("required_args", "cwd_exists", "ref_syntax", "change_scope", "privilege"),
        ("command_available",),
    ),
    Family.TEST: (
        "execute",
        "process",
        ("required_args", "cwd_exists", "privilege"),
        ("test_invocation", "command_available"),
    ),
    Family.SEARCH: ("query", "none", ("required_args", "arg_types"), ("search_root",)),
    Family.HTTP: ("network", "network", ("required_args", "url_syntax"), ()),
    Family.UNKNOWN: ("unknown", "unknown", (), ()),
}

#: Documented required arguments of host built-in tools (the host owns the
#: schema; Headroom never rewrites it).
_REQUIRED: dict[str, tuple[tuple[str, ...], dict[str, str]]] = {
    "read": (("file_path",), {"file_path": "string", "offset": "integer", "limit": "integer"}),
    "edit": (
        ("file_path", "old_string", "new_string"),
        {
            "file_path": "string",
            "old_string": "string",
            "new_string": "string",
            "replace_all": "boolean",
        },
    ),
    "write": (("file_path", "content"), {"file_path": "string", "content": "string"}),
    "multiedit": (("file_path", "edits"), {"file_path": "string", "edits": "array"}),
    "notebookedit": (("notebook_path",), {"notebook_path": "string"}),
    "bash": (
        ("command",),
        {"command": "string", "timeout": "integer", "run_in_background": "boolean"},
    ),
    "powershell": (("command",), {"command": "string", "timeout": "integer"}),
    "grep": (
        ("pattern",),
        {"pattern": "string", "path": "string", "glob": "string", "head_limit": "integer"},
    ),
    "glob": (("pattern",), {"pattern": "string", "path": "string"}),
    "webfetch": (("url",), {"url": "string"}),
    "websearch": (("query",), {"query": "string"}),
    "apply_patch": ((), {"input": "string"}),
}
_ENUMS: dict[str, dict[str, tuple[str, ...]]] = {
    "grep": {"output_mode": ("content", "files_with_matches", "count")},
}
_EXCLUSIVE: dict[str, tuple[tuple[str, str], ...]] = {
    "grep": (("glob", "type"),),
}
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
    "null": (type(None),),
}

# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------

_POSIX_BUILTINS = frozenset(
    "cd echo export set unset source . test [ [[ printf read pwd exit true false type alias "
    "unalias eval exec shift trap wait ulimit umask command builtin local declare typeset "
    "readonly return break continue let hash getopts jobs fg bg kill disown history pushd "
    "popd dirs shopt time times compgen complete enable help logout mapfile readarray suspend "
    ": if then else elif fi for while until do done case esac function select coproc".split()
)
_PS_ALIASES = frozenset(
    "cd ls dir cat type echo write cp copy mv move rm del erase rmdir rd mkdir md pwd cls clear "
    "gci gc sc ac ni ri mi cpi sl gl iex iwr irm sls select where foreach % ? ps kill sort "
    "measure tee start sleep set gv sv gal".split()
)
_PS_CMDLET = re.compile(r"^[A-Za-z]+-[A-Za-z][A-Za-z0-9]*$")
_ESCALATION = re.compile(
    r"(?:^|[\s;&|(])(?:sudo|doas|su\s+-|pkexec|runas(?:\.exe)?)\b|-Verb\s+RunAs\b|"
    r"\bchmod\s+(?:-R\s+)?(?:777|a\+rwx)\s+/(?:\s|$)|\bchown\s+-R\s+root\b",
    re.IGNORECASE,
)
_ESCALATION_ASK = re.compile(
    r"\b(?:sudo|root|admin(?:istrator)?|elevat\w*|runas|privileg\w*)\b", re.I
)
_CMD_SWITCH_TOOLS = frozenset(
    {"dir", "del", "copy", "xcopy", "findstr", "move", "rd", "rmdir", "attrib", "type"}
)
_REF_COMMANDS = frozenset(
    {
        "checkout",
        "switch",
        "branch",
        "rebase",
        "merge",
        "reset",
        "tag",
        "cherry-pick",
        "revert",
        "show",
        "log",
        "diff",
    }
)
_TESTPATH = re.compile(r"^[\w./\\-]+\.(?:py|rs|js|ts|tsx|jsx|go)(?:::[\w\[\]:.-]+)?$")
_REPAIRABLE_CATEGORIES = frozenset(
    {
        "file_not_found",
        "command_not_found",
        "module_not_found",
        "exit_code",
        "no_matches",
        "runtime_error",
        "syntax_error",
    }
)


def _command(args: Mapping[str, Any]) -> str:
    command = args.get("command") or args.get("cmd")
    if isinstance(command, list):
        return " ".join(str(c) for c in command)
    return command if isinstance(command, str) else ""


def _first_word(command: str) -> str:
    for segment in command_segments(command):
        words = split_command(segment)
        while words and (
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0])
            or words[0] in ("sudo", "time", "env", "nohup", "command")
        ):
            words = words[1:]
        if not words:
            continue
        if words[0] in ("cd", "pushd", "set-location", "sl"):
            continue
        return words[0].lower().rsplit("/", 1)[-1].removesuffix(".exe")
    return ""


def check_ref_format(ref: str) -> bool:
    """``git check-ref-format --allow-onelevel`` rules (plus rev suffixes)."""
    base = re.split(r"[~^]", ref, maxsplit=1)[0] if not ref.startswith(("~", "^")) else ref
    if base in ("HEAD", "@", "-", "FETCH_HEAD", "ORIG_HEAD", "MERGE_HEAD"):
        return True
    if "@{" in base:
        base = base.split("@{", 1)[0] or "HEAD"
    if not base or base.startswith(("-", "/")) or base.endswith(("/", ".", ".lock")):
        return False
    if ".." in base or "//" in base or "/." in base or base.startswith("."):
        return False
    if any(ch in base for ch in " ~^:?*[\\") or any(ord(ch) < 32 or ord(ch) == 127 for ch in base):
        return False
    return True


def _levenshtein(a: str, b: str) -> int:
    from headroom.memory.traffic_learner import _levenshtein as lev

    return lev(a, b)


def _typo_candidates(
    name: str, options: Iterable[str], *, limit: int = 3, strict: bool = False
) -> list[str]:
    """Close spellings of ``name`` among ``options`` (TrafficLearner's typo rule).

    ``strict`` (command names) only accepts a transposition or one edit, so
    short unrelated tools (``pip``/``git``) never pair.
    """
    from headroom.memory.traffic_learner import _paths_related_as_typo

    low = name.lower()
    scored: list[tuple[int, str]] = []
    for option in options:
        if option == name:
            continue
        olow = option.lower()
        distance = 0 if olow == low else _levenshtein(low, olow)
        if strict:
            transposed = len(low) == len(olow) and sorted(low) == sorted(olow) and distance <= 2
            related = olow == low or transposed or distance <= (1 if len(low) < 6 else 2)
        else:
            related = olow == low or _paths_related_as_typo(name, option)
        if related:
            scored.append((distance, option))
    scored.sort()
    return [o for _, o in scored[:limit]]


# ---------------------------------------------------------------------------
# Learned recovery (TrafficLearner pairing + classify_error)
# ---------------------------------------------------------------------------


def environment_fingerprint(shell: str = "posix") -> str:
    from headroom.agent.change_scope import _digest

    return _digest(
        sys.platform, os.name, shell, os.environ.get("SHELL", "") or os.environ.get("ComSpec", "")
    )


@dataclass
class Recovery:
    key: str
    tool: str
    failed: str
    success: str
    error_category: str
    fingerprint: str
    corroborations: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    promoted: bool = False
    invalidated: bool = False
    call_ids: set[str] = field(default_factory=set)

    def live(self, now: float, fingerprint: str) -> bool:
        return (
            self.promoted
            and not self.invalidated
            and self.fingerprint == fingerprint
            and now - self.last_seen <= RECOVERY_TTL_S
        )


class RecoveryLedger:
    """Workspace-scoped learned failure→recovery suggestions."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._history: OrderedDict[str, deque[dict[str, Any]]] = OrderedDict()
        self._recoveries: OrderedDict[tuple[str, str], Recovery] = OrderedDict()
        self.counters: dict[str, int] = {
            "observed": 0,
            "corroborated": 0,
            "promoted": 0,
            "invalidated": 0,
            "expired": 0,
        }

    @staticmethod
    def _learner_name(tool_name: str) -> str:
        from headroom.learn._shared import normalize_tool_name

        lname = tool_name.lower()
        if lname in _EXEC:
            return "Bash"
        return (
            normalize_tool_name(tool_name)
            if lname not in ("read", "grep", "glob")
            else tool_name.capitalize()
        )

    @staticmethod
    def _key_text(tool: str, value: str) -> str:
        from headroom.memory.traffic_learner import _normalize_bash_for_hash

        if tool == "Bash":
            return _normalize_bash_for_hash(value)
        return value.replace("\\", "/")

    def observe(
        self,
        workspace_id: str,
        session_id: str,
        tool_name: str,
        tool_input: Mapping[str, Any],
        output: str,
        is_error: bool,
        *,
        call_id: str = "",
        shell: str = "posix",
    ) -> Recovery | None:
        """Record an executed call; returns a recovery it corroborated, if any."""
        from headroom.agent.enforcement import _extractor
        from headroom.learn._shared import classify_error

        tool = self._learner_name(tool_name)
        entry = {
            "tool_name": tool,
            "input": dict(tool_input),
            "output": output[:2000],
            "is_error": is_error,
            "error_category": classify_error(output).value if is_error else None,
        }
        now = self._clock()
        fingerprint = environment_fingerprint(shell)
        corroborated: Recovery | None = None
        with self._lock:
            self.counters["observed"] += 1
            history = self._history.setdefault(session_id, deque(maxlen=_HISTORY_LIMIT))
            self._history.move_to_end(session_id)
            while len(self._history) > _SESSIONS_LIMIT:
                self._history.popitem(last=False)
            past = list(history)
        if is_error:
            self._contradict(workspace_id, tool, tool_input, success_failed=True)
        else:
            self._contradict(workspace_id, tool, tool_input, success_failed=False)
            for prev in reversed(past[-5:]):
                if not prev["is_error"] or prev["tool_name"] != tool:
                    continue
                if prev.get("error_category") not in _REPAIRABLE_CATEGORIES:
                    break
                pattern = _extractor()._build_recovery_pattern(prev, entry)
                if pattern is None:
                    break
                meta = pattern.metadata
                failed = str(meta.get("failed_cmd") or meta.get("error_path") or "")
                success = str(meta.get("success_cmd") or meta.get("success_path") or "")
                if not failed or not success:
                    break
                corroborated = self._corroborate(
                    workspace_id,
                    pattern.content_hash,
                    tool,
                    failed,
                    success,
                    str(prev.get("error_category") or "unknown"),
                    fingerprint,
                    now,
                    call_id,
                )
                break
        with self._lock:
            history.append(entry)
        return corroborated

    def _corroborate(
        self,
        workspace_id: str,
        key: str,
        tool: str,
        failed: str,
        success: str,
        category: str,
        fingerprint: str,
        now: float,
        call_id: str,
    ) -> Recovery:
        with self._lock:
            rec = self._recoveries.get((workspace_id, key))
            if rec is None or rec.fingerprint != fingerprint:
                rec = Recovery(key, tool, failed, success, category, fingerprint, first_seen=now)
                self._recoveries[(workspace_id, key)] = rec
            if call_id and call_id in rec.call_ids:
                return rec  # the same recovery replayed in a later request
            if call_id:
                rec.call_ids.add(call_id)
            rec.corroborations += 1
            rec.last_seen = now
            rec.success = success
            self.counters["corroborated"] += 1
            if (
                not rec.promoted
                and not rec.invalidated
                and rec.corroborations >= PROMOTION_THRESHOLD
            ):
                rec.promoted = True
                self.counters["promoted"] += 1
            self._recoveries.move_to_end((workspace_id, key))
            while len(self._recoveries) > _RECOVERIES_LIMIT:
                self._recoveries.popitem(last=False)
            return rec

    def _contradict(
        self, workspace_id: str, tool: str, tool_input: Mapping[str, Any], *, success_failed: bool
    ) -> None:
        value = (
            _command(tool_input)
            if tool == "Bash"
            else str(tool_input.get("file_path") or tool_input.get("path") or "")
        )
        if not value:
            return
        text = self._key_text(tool, value)
        with self._lock:
            for (ws, _), rec in self._recoveries.items():
                if ws != workspace_id or rec.tool != tool or rec.invalidated:
                    continue
                # The suggested recovery failing, or the "failing" form now
                # succeeding, contradicts the learned pattern.
                target = rec.success if success_failed else rec.failed
                if self._key_text(tool, target) == text:
                    rec.invalidated = True
                    self.counters["invalidated"] += 1

    def suggest(
        self,
        workspace_id: str,
        tool_name: str,
        tool_input: Mapping[str, Any],
        *,
        shell: str = "posix",
    ) -> list[str]:
        tool = self._learner_name(tool_name)
        value = (
            _command(tool_input)
            if tool == "Bash"
            else str(tool_input.get("file_path") or tool_input.get("path") or "")
        )
        if not value:
            return []
        text = self._key_text(tool, value)
        now = self._clock()
        fingerprint = environment_fingerprint(shell)
        out: list[str] = []
        with self._lock:
            for (ws, _), rec in self._recoveries.items():
                if ws != workspace_id or rec.tool != tool:
                    continue
                if rec.promoted and now - rec.last_seen > RECOVERY_TTL_S and not rec.invalidated:
                    rec.invalidated = True
                    self.counters["expired"] += 1
                if rec.live(now, fingerprint) and self._key_text(tool, rec.failed) == text:
                    out.append(rec.success)
        return out

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "recoveries": len(self._recoveries),
                "promoted_live": sum(
                    1 for r in self._recoveries.values() if r.promoted and not r.invalidated
                ),
                "counters": dict(self.counters),
            }


# ---------------------------------------------------------------------------
# Validator service
# ---------------------------------------------------------------------------


def _schemas_from_tools(tools: Any) -> dict[str, Mapping[str, Any]]:
    """``name -> JSON schema`` from Anthropic / OpenAI tools arrays (read-only)."""
    out: dict[str, Mapping[str, Any]] = {}
    if not isinstance(tools, list):
        return out
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        if isinstance(tool.get("function"), Mapping):
            fn = tool["function"]
            name, schema = fn.get("name"), fn.get("parameters")
        else:
            name, schema = tool.get("name"), tool.get("input_schema")
        if isinstance(name, str) and isinstance(schema, Mapping):
            out[name] = schema
    return out


def validate_schema(args: Mapping[str, Any], schema: Mapping[str, Any]) -> list[Finding]:
    """Subset JSON-schema check: object type, required, property types, enums."""
    findings: list[Finding] = []
    if schema.get("type") not in (None, "object"):
        return findings
    props = schema.get("properties") if isinstance(schema.get("properties"), Mapping) else {}
    for name in schema.get("required") or ():
        if name not in args:
            findings.append(
                Finding(
                    "contract.missing_required_arg",
                    "hard",
                    f"required argument `{name}` is missing",
                )
            )
    for name, value in args.items():
        spec = props.get(name) if isinstance(props, Mapping) else None
        if not isinstance(spec, Mapping):
            if schema.get("additionalProperties") is False:
                findings.append(
                    Finding(
                        "contract.unknown_arg",
                        "advisory",
                        f"argument `{name}` is not in the tool schema",
                    )
                )
            continue
        types = spec.get("type")
        allowed = types if isinstance(types, list) else [types] if isinstance(types, str) else []
        if allowed:
            ok = any(
                isinstance(value, _JSON_TYPES.get(t, (object,)))
                and not (t in ("integer", "number") and isinstance(value, bool))
                for t in allowed
            )
            if not ok:
                findings.append(
                    Finding(
                        "contract.arg_type",
                        "hard",
                        f"argument `{name}` must be {'/'.join(allowed)}",
                    )
                )
        enum = spec.get("enum")
        if isinstance(enum, list) and value not in enum:
            findings.append(
                Finding("contract.arg_enum", "hard", f"argument `{name}` must be one of {enum[:6]}")
            )
    return findings


class ToolContractValidator:
    """Owner of tool contracts and argument validation (see module docstring)."""

    def __init__(
        self,
        runtime: Any = None,
        *,
        firewall: Any = None,
        policy: Any = None,
        clock: Callable[[], float] = time.time,
        which: Callable[[str], str | None] = shutil.which,
    ) -> None:
        from headroom.agent.change_scope import EnforcementPolicy

        self.runtime = runtime
        self.firewall = firewall
        self.policy = policy or EnforcementPolicy.from_env()
        self._clock = clock
        self._which = which
        self.recoveries = RecoveryLedger(clock)
        self._lock = threading.Lock()
        self._which_cache: OrderedDict[str, bool] = OrderedDict()
        self.counters: dict[str, int] = {
            "evaluated": 0,
            "allowed": 0,
            "warned": 0,
            "repair_suggested": 0,
            "prevented": 0,
            "unknown_tools_observed": 0,
            "observed": 0,
            "observed_invalid": 0,
            "observed_failures": 0,
            "jevk5_ranked": 0,
        }
        self.reasons: dict[str, int] = {}

    # -- contracts -----------------------------------------------------------

    def contract_for(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        *,
        source: str = "claude",
        workspace_id: str = "",
    ) -> ToolContract:
        family = family_of(tool_name, args)
        op, effect, hard, advisory = _CONTRACTS[family]
        learned = (
            tuple(self.recoveries.suggest(workspace_id, tool_name, args, shell=shell_of(tool_name)))
            if workspace_id
            else ()
        )
        return ToolContract(
            tool_family=family,
            source=source,
            operation_class=op,
            side_effect=effect,
            schema_version=SCHEMA_VERSION,
            hard_rules=hard,
            advisory_rules=advisory,
            learned_recovery=learned,
            tool_name=tool_name,
            certified=family is not Family.UNKNOWN,
        )

    # -- checks --------------------------------------------------------------

    def _available(self, name: str) -> bool:
        with self._lock:
            if name in self._which_cache:
                return self._which_cache[name]
        found = bool(self._which(name))
        with self._lock:
            self._which_cache[name] = found
            while len(self._which_cache) > 512:
                self._which_cache.popitem(last=False)
        return found

    def _schema_findings(
        self, tool_name: str, args: Mapping[str, Any], schema: Mapping[str, Any] | None
    ) -> list[Finding]:
        if schema is not None:
            return validate_schema(args, schema)
        lname = tool_name.lower()
        spec = _REQUIRED.get(lname)
        findings: list[Finding] = []
        if spec is None:
            return findings
        required, types = spec
        findings.extend(
            validate_schema(
                args,
                {
                    "type": "object",
                    "required": list(required),
                    "properties": {k: {"type": v} for k, v in types.items()},
                },
            )
        )
        for name, values in _ENUMS.get(lname, {}).items():
            if name in args and args[name] not in values:
                findings.append(
                    Finding(
                        "contract.arg_enum",
                        "hard",
                        f"argument `{name}` must be one of {list(values)}",
                    )
                )
        for a, b in _EXCLUSIVE.get(lname, ()):
            if args.get(a) not in (None, "") and args.get(b) not in (None, ""):
                findings.append(
                    Finding(
                        "contract.mutually_exclusive",
                        "advisory",
                        f"`{a}` and `{b}` are mutually exclusive",
                    )
                )
        if (
            lname == "edit"
            and isinstance(args.get("old_string"), str)
            and args.get("old_string") == args.get("new_string")
        ):
            findings.append(
                Finding("contract.noop_edit", "hard", "old_string and new_string are identical")
            )
        return findings

    def _path_findings(
        self,
        family: Family,
        tool_name: str,
        args: Mapping[str, Any],
        identity: Any,
        cwd: str | None,
    ) -> tuple[list[Finding], list[str]]:
        findings: list[Finding] = []
        repairs: list[str] = []
        lname = tool_name.lower()
        raw = None
        for key in ("file_path", "path", "notebook_path", "filePath", "target_file"):
            if isinstance(args.get(key), str) and args[key]:
                raw = args[key]
                break
        if raw is None:
            return findings, repairs
        root = identity.root if identity.can_authorize_paths() else None
        canon = canonicalize(raw, root=root, cwd=cwd or root)
        if family is Family.EDIT and canon.unc:
            findings.append(
                Finding("contract.unc_target", "advisory", "the target is a UNC network path")
            )
        must_exist = family is Family.READ or lname in (
            "edit",
            "multiedit",
            "notebookedit",
            "str_replace_editor",
            "edit_file",
            "replace_in_file",
        )
        if family is Family.SEARCH:
            must_exist = True
        if must_exist and not canon.exists:
            directory = os.path.dirname(canon.resolved)
            name = os.path.basename(canon.resolved)
            options: list[str] = []
            try:
                if os.path.isdir(directory):
                    with os.scandir(directory) as it:
                        for i, entry in enumerate(it):
                            if i >= _TYPO_DIR_LIMIT:
                                break
                            options.append(entry.name)
            except OSError:
                options = []
            candidates = _typo_candidates(name, options)
            for candidate in candidates:
                full = os.path.join(directory, candidate).replace("\\", "/")
                # Only suggest a repair that is itself in scope-safe territory.
                if root is not None:
                    alt = canonicalize(full, root=root)
                    if not alt.inside:
                        continue
                repairs.append(full)
            code = "contract.input_missing"
            findings.append(Finding(code, "advisory", f"`{raw}` does not exist"))
        return findings, repairs

    def _exec_findings(
        self,
        family: Family,
        tool_name: str,
        args: Mapping[str, Any],
        identity: Any,
        cwd: str | None,
        user_text: str,
    ) -> tuple[list[Finding], list[str]]:
        findings: list[Finding] = []
        repairs: list[str] = []
        command = _command(args)
        if not command.strip():
            return findings, repairs
        shell = shell_of(tool_name)
        if cwd and not os.path.isdir(cwd):
            findings.append(
                Finding("contract.cwd_missing", "hard", f"working directory `{cwd}` does not exist")
            )
        target = shell_cwd(command, cwd, shell)
        if target and target != cwd and not os.path.isdir(target):
            findings.append(
                Finding(
                    "contract.cwd_missing",
                    "hard",
                    f"`cd {os.path.basename(target)}` target does not exist",
                )
            )
        if _ESCALATION.search(command) and not _ESCALATION_ASK.search(user_text):
            findings.append(
                Finding(
                    "contract.privilege_escalation",
                    "hard",
                    "elevates privileges the user did not ask for",
                )
            )
        for segment in command_segments(command)[:12]:
            words = split_command(segment, shell)
            while words and (
                re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0])
                or words[0] in ("time", "nohup", "env", "command", "sudo", "doas")
            ):
                words = words[1:]
            if not words:
                continue
            head = words[0]
            base = head.lower().rsplit("/", 1)[-1].rsplit("\\", 1)[-1].removesuffix(".exe")
            if (
                shell == "posix"
                and base in _CMD_SWITCH_TOOLS
                and any(re.match(r"^/[A-Za-z?]$", w) for w in words[1:])
            ):
                findings.append(
                    Finding(
                        "contract.platform_switch",
                        "advisory",
                        f"`{base} /X` is cmd.exe syntax; this is a POSIX shell",
                    )
                )
            if (
                shell == "powershell"
                and base in ("rm", "del", "ri", "remove-item")
                and any(re.match(r"^-[rf]{2}$", w) for w in words[1:])
            ):
                findings.append(
                    Finding(
                        "contract.platform_switch",
                        "advisory",
                        "PowerShell's rm has no -rf; use Remove-Item -Recurse -Force",
                    )
                )
            known = (
                base in _POSIX_BUILTINS
                or (shell == "powershell" and (base in _PS_ALIASES or _PS_CMDLET.match(head)))
                or head.startswith(("./", ".\\", "/", "~", "$", "%"))
                or re.match(r"^[A-Za-z]:[\\/]", head)
                or "=" in head
                or head.startswith(("(", "{", "'", '"', "#"))
            )
            if known:
                pass
            elif not self._available(head):
                options = [
                    c for c in _typo_candidates(base, _COMMON_COMMANDS) if self._available(c)
                ]
                if options:
                    repairs.extend(
                        re.sub(r"^" + re.escape(head), o, segment, count=1) for o in options
                    )
                findings.append(
                    Finding("contract.command_unavailable", "advisory", f"`{base}` is not on PATH")
                )
            if base == "git":
                findings.extend(_git_findings(words))
            if family is Family.TEST:
                findings.extend(_test_findings(base, words, target or cwd))
        return findings, repairs

    # -- evaluation ----------------------------------------------------------

    def evaluate(
        self,
        tool_name: str,
        tool_input: Mapping[str, Any],
        *,
        identity: Any,
        source: str = "claude",
        cwd: str | None = None,
        enforcement: str = "observe_only",
        call_id: str = "",
        schema: Mapping[str, Any] | None = None,
    ) -> Decision:
        from headroom.agent.change_scope import Enforcement
        from headroom.agent.enforcement import HOST_PRETOOL, scope_decisions

        args = dict(tool_input) if isinstance(tool_input, Mapping) else {}
        workspace = getattr(identity, "workspace_id", "")
        contract = self.contract_for(tool_name, args, source=source, workspace_id=workspace)
        enforceable = enforcement == HOST_PRETOOL
        if contract.tool_family is Family.UNKNOWN:
            self._bump("unknown_tools_observed")
            return Decision(
                DecisionKind.ALLOW,
                "contract.unknown_tool",
                args,
                0,
                0,
                contract,
                enforcement=enforcement,
            )
        call_cwd = cwd or (identity.root if identity.can_authorize_paths() else None)
        findings = self._schema_findings(tool_name, args, schema)
        repairs: list[str] = []
        user_text = self._user_text(identity)
        if contract.tool_family in (Family.READ, Family.EDIT, Family.SEARCH):
            more, fixes = self._path_findings(
                contract.tool_family, tool_name, args, identity, call_cwd
            )
            findings += more
            repairs += fixes
        if contract.tool_family in (Family.EXEC, Family.GIT, Family.TEST):
            more, fixes = self._exec_findings(
                contract.tool_family, tool_name, args, identity, call_cwd, user_text
            )
            findings += more
            repairs += fixes
        if contract.tool_family is Family.HTTP and isinstance(args.get("url"), str):
            if not re.match(r"^https?://[^\s/]+", args["url"]):
                findings.append(
                    Finding("contract.url_syntax", "hard", "url must be an absolute http(s) URL")
                )
        for learned in contract.learned_recovery:
            if learned not in repairs:
                repairs.append(learned)
        if repairs and not any(
            f.code in ("contract.input_missing", "contract.command_unavailable") for f in findings
        ):
            findings.append(
                Finding(
                    "contract.learned_failure",
                    "advisory",
                    "this call matches a recurring failure in this workspace",
                )
            )
        scope: list[ScopeDecision] = []
        if self.firewall is not None and contract.side_effect in ("filesystem", "process", "vcs"):
            scope = scope_decisions(
                self.firewall, identity, tool_name, args, cwd=call_cwd, enforceable=enforceable
            )
        repairs = self._rank(repairs, tool_name, args)
        kind, reason = self._decide(findings, repairs, scope, enforceable)
        mode = self.policy.tool_contracts
        if mode is Enforcement.OBSERVE and kind is not DecisionKind.BLOCK:
            kind_out = DecisionKind.ALLOW
        else:
            kind_out = kind
        if kind_out is DecisionKind.BLOCK and not enforceable:
            kind_out = DecisionKind.WARN
        task_rev = max((s.task_revision for s in scope), default=self._task_revision(identity))
        scope_rev = max((s.scope_revision for s in scope), default=self._scope_revision(identity))
        decision = Decision(
            kind_out,
            reason,
            args,
            task_rev,
            scope_rev,
            contract,
            findings=tuple(findings),
            repairs=tuple(repairs),
            scope=tuple(scope),
            enforcement=enforcement,
        )
        self._record(decision, identity, call_id, observed=kind if kind_out is not kind else None)
        return decision

    def _decide(
        self,
        findings: Sequence[Finding],
        repairs: Sequence[str],
        scope: Sequence[ScopeDecision],
        enforceable: bool,
    ) -> tuple[DecisionKind, str]:
        from headroom.agent.change_scope import Enforcement

        blocked = [s for s in scope if s.verdict is Verdict.BLOCK]
        if blocked:
            return DecisionKind.BLOCK, blocked[0].reason_code
        hard = [f for f in findings if f.severity == "hard"]
        escalation = [f for f in hard if f.code == "contract.privilege_escalation"]
        if escalation and enforceable:
            return DecisionKind.BLOCK, escalation[0].code
        if hard and self.policy.tool_contracts is Enforcement.PROTECT and enforceable:
            return DecisionKind.BLOCK, hard[0].code
        if hard:
            return DecisionKind.WARN, hard[0].code
        if repairs:
            return DecisionKind.REPAIR_SUGGEST, (
                findings[0].code if findings else "contract.learned_failure"
            )
        warned_scope = [
            s
            for s in scope
            if s.verdict is Verdict.WARN and s.reason_code != "scope.untrusted_workspace"
        ]
        if warned_scope:
            return DecisionKind.WARN, warned_scope[0].reason_code
        if findings:
            return DecisionKind.WARN, findings[0].code
        return DecisionKind.ALLOW, "contract.ok"

    def _rank(self, repairs: list[str], tool_name: str, args: Mapping[str, Any]) -> list[str]:
        """JevK5 may reorder already-safe repairs; it can never add one."""
        unique = list(dict.fromkeys(repairs))
        runtime = self.runtime
        advisor = getattr(runtime, "advisor", None) if runtime is not None else None
        if len(unique) < 2 or advisor is None:
            return unique
        options = {f"r{i}": r[:200] for i, r in enumerate(unique[:8])}
        try:
            advice = advisor.choose(
                "tool_repair_rank",
                {"tool": tool_name, "call": str(dict(args))[:400], "options": options},
                "Which suggested replacement most likely matches what the call intended?",
                options,
                budget=advisor.request_budget(),
            )
        except Exception as exc:  # noqa: BLE001 - advisory only
            log.debug("repair ranking advisor failed: %s", describe_exception(exc))
            return unique
        if advice is None or str(advice.value) not in options:
            return unique
        chosen = unique[int(str(advice.value)[1:])]
        self._bump("jevk5_ranked")
        return [chosen] + [r for r in unique if r != chosen]

    # -- revisions / state ---------------------------------------------------

    def _snapshot(self, identity: Any) -> Any:
        runtime = self.runtime
        compiler = runtime.services.get("task_state") if runtime is not None else None
        if compiler is None or not identity.trusted:
            return None
        return compiler.snapshot(identity.workspace_id, identity.session_id)

    def _task_revision(self, identity: Any) -> int:
        snap = self._snapshot(identity)
        return snap.state.revision if snap is not None else 0

    def _scope_revision(self, identity: Any) -> int:
        if self.firewall is None or not identity.can_authorize_paths():
            return 0
        return int(self.firewall.contract(identity).revision)

    def _user_text(self, identity: Any) -> str:
        snap = self._snapshot(identity)
        if snap is None:
            return ""
        return " ".join(a.text for a in snap.state.atoms if a.origin == "user" and a.live)[:4000]

    # -- recording -----------------------------------------------------------

    def _bump(self, name: str) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + 1

    def _record(
        self, decision: Decision, identity: Any, call_id: str, *, observed: DecisionKind | None
    ) -> None:
        name = {
            DecisionKind.BLOCK: "prevented",
            DecisionKind.WARN: "warned",
            DecisionKind.REPAIR_SUGGEST: "repair_suggested",
        }.get(decision.kind, "allowed")
        with self._lock:
            self.counters["evaluated"] += 1
            self.counters[name] += 1
            if decision.reason_code != "contract.ok":
                self.reasons[decision.reason_code] = self.reasons.get(decision.reason_code, 0) + 1
                while len(self.reasons) > 256:
                    self.reasons.pop(next(iter(self.reasons)))
        runtime = self.runtime
        if runtime is None or not getattr(identity, "workspace_id", ""):
            return
        meta = {
            "kind": decision.kind.value,
            "observed_kind": observed.value if observed is not None else None,
            "reason_code": decision.reason_code,
            "family": decision.contract.tool_family.value,
            "certified": decision.contract.certified,
            "enforcement": decision.enforcement,
            "task_revision": decision.task_revision,
            "scope_revision": decision.scope_revision,
            "findings": [f.code for f in decision.findings][:8],
            "repairs": len(decision.repairs),
        }
        try:
            event = make_event(
                workspace_id=identity.workspace_id,
                session_id=identity.session_id,
                kind=EventKind.CONTRACT_DECISION,
                origin=Origin.SYSTEM,
                source_id=f"contract:{call_id or decision.contract.tool_name}",
                index=decision.scope_revision,
                payload=meta,
                revision=decision.scope_revision,
                metadata=meta,
            )
            runtime.submit([EventEnvelope(event=event)])
        except Exception as exc:  # noqa: BLE001 - audit is best-effort
            log.debug("contract decision event failed: %s", describe_exception(exc))

    def observe(
        self,
        tool_name: str,
        tool_input: Mapping[str, Any],
        *,
        identity: Any,
        call_id: str = "",
        output: str = "",
        is_error: bool = False,
        evaluated_at_host: bool = False,
        tools: Any = None,
    ) -> None:
        """Proxy path (after execution): advisory validation + learned recovery."""
        args = dict(tool_input) if isinstance(tool_input, Mapping) else {}
        self._bump("observed")
        if is_error:
            self._bump("observed_failures")
        if not evaluated_at_host:
            schema = _schemas_from_tools(tools).get(tool_name)
            if schema is not None and validate_schema(args, schema):
                self._bump("observed_invalid")
        if identity.trusted and family_of(tool_name, args) is not Family.UNKNOWN:
            self.recoveries.observe(
                identity.workspace_id,
                identity.session_id,
                tool_name,
                args,
                output,
                is_error,
                call_id=call_id,
                shell=shell_of(tool_name),
            )

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "policy": self.policy.to_dict(),
                "counters": dict(self.counters),
                "reasons": dict(sorted(self.reasons.items(), key=lambda kv: -kv[1])[:16]),
                "recovery": self.recoveries.status(),
                "promotion_threshold": PROMOTION_THRESHOLD,
            }


def _git_findings(words: list[str]) -> list[Finding]:
    args = words[1:]
    while args and args[0] in ("-C", "-c"):
        args = args[2:]
    if not args:
        return []
    sub, rest = args[0], args[1:]
    if sub not in _REF_COMMANDS:
        return []
    if "--" in rest:
        rest = rest[: rest.index("--")]
    out: list[Finding] = []
    skip_next = False
    for arg in rest:
        if skip_next:
            skip_next = False
            continue
        if arg in ("-b", "-B", "-c", "-C", "-m", "--onto", "-t", "--track"):
            skip_next = arg in ("-m",)
            continue
        if arg.startswith("-") or "=" in arg:
            continue
        if sub in ("log", "diff", "show") and ("/" in arg or "." in arg) and os.path.exists(arg):
            continue
        refs = (
            arg.split("...")
            if "..." in arg
            else arg.split("..")
            if ".." in arg and sub in ("log", "diff")
            else [arg]
        )
        for ref in refs:
            if ref and not check_ref_format(ref):
                out.append(
                    Finding(
                        "contract.git_ref_syntax", "hard", f"`{ref}` is not a valid git ref name"
                    )
                )
                break
    return out


def _test_findings(base: str, words: list[str], cwd: str | None) -> list[Finding]:
    out: list[Finding] = []
    here = cwd or "."
    if base in ("pytest", "py.test") or (base.startswith("python") and "pytest" in words):
        args = words[words.index("pytest") + 1 :] if "pytest" in words else words[1:]
        for i, arg in enumerate(args):
            if arg in ("-k", "-m") and (i + 1 >= len(args) or args[i + 1].startswith("-")):
                out.append(
                    Finding("contract.test_invocation", "hard", f"`{arg}` needs an expression")
                )
            if _TESTPATH.match(arg):
                path = arg.split("::", 1)[0]
                if not os.path.exists(os.path.join(here, path)):
                    out.append(
                        Finding(
                            "contract.test_path_missing",
                            "advisory",
                            f"test path `{path}` does not exist",
                        )
                    )
    elif base in ("npm", "yarn", "pnpm", "bun") and not os.path.exists(
        os.path.join(here, "package.json")
    ):
        out.append(
            Finding(
                "contract.test_invocation", "advisory", "no package.json in the working directory"
            )
        )
    elif base == "cargo" and not os.path.exists(os.path.join(here, "Cargo.toml")):
        found = any(os.path.exists(os.path.join(p, "Cargo.toml")) for p in _parents(here))
        if not found:
            out.append(
                Finding(
                    "contract.test_invocation",
                    "advisory",
                    "no Cargo.toml in or above the working directory",
                )
            )
    return out


def _parents(path: str) -> list[str]:
    out: list[str] = []
    current = os.path.abspath(path)
    for _ in range(12):
        parent = os.path.dirname(current)
        if parent == current:
            break
        out.append(parent)
        current = parent
    return out


#: Commands a typo'd command name is compared against (only suggested when
#: actually available on PATH).
_COMMON_COMMANDS = (
    "git",
    "python",
    "python3",
    "pip",
    "pip3",
    "uv",
    "pytest",
    "node",
    "npm",
    "npx",
    "pnpm",
    "yarn",
    "bun",
    "cargo",
    "rustc",
    "go",
    "make",
    "cmake",
    "docker",
    "kubectl",
    "ls",
    "cat",
    "grep",
    "rg",
    "find",
    "sed",
    "awk",
    "curl",
    "wget",
    "jq",
    "mvn",
    "gradle",
    "dotnet",
    "java",
    "javac",
    "ruby",
    "bundle",
    "php",
    "composer",
    "deno",
    "tsc",
    "eslint",
    "prettier",
    "ruff",
    "black",
    "mypy",
    "tox",
    "nox",
    "poetry",
    "gh",
    "head",
    "tail",
    "sort",
    "diff",
    "tar",
    "unzip",
)

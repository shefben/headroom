"""Secret scrubbing and bounded text helpers for persisted agent state.

Task-state atoms and evidence facts persist short, derived text (a goal, an
error signature). Before anything is stored it passes through
:func:`scrub_secrets`, which replaces credential-shaped substrings, and is
bounded by :func:`clip`. Entire files and raw tool output are never stored.
"""

from __future__ import annotations

import re

REDACTED = "[redacted]"

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[abposr]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,}"),
    re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/\-]{12,}=*"),
    re.compile(
        r"(?i)\b((?:api[_-]?key|access[_-]?key|secret|token|password|passwd|pwd|auth)"
        r"[A-Za-z0-9_]*)\s*[=:]\s*['\"]?[^\s'\"]{4,}"
    ),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)"),
    re.compile(r"\b[A-Fa-f0-9]{40,}\b"),
    re.compile(r"\b[A-Za-z0-9+/]{48,}={0,2}"),
)


def scrub_secrets(text: str) -> str:
    """Replace credential-shaped substrings with ``[redacted]``."""
    out = text
    for pattern in _SECRET_PATTERNS:
        if pattern.groups and pattern.pattern.startswith("(?i)\\b(("):
            out = pattern.sub(lambda m: f"{m.group(1)}={REDACTED}", out)
        else:
            out = pattern.sub(REDACTED, out)
    return out


def clip(text: str, limit: int) -> str:
    """Collapse whitespace and bound ``text`` to ``limit`` characters."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - 1)].rstrip() + "…"


def safe_text(text: str, limit: int) -> str:
    """Scrub then clip: the only form of free text agent state persists."""
    return clip(scrub_secrets(text), limit)

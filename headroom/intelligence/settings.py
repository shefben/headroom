"""Typed configuration for the local JevK5 decision advisor.

One frozen object holds every tunable with its tested bounds. Values come from
the existing configuration mechanism — environment variables, which
``settings.json`` feeds through ``settings_store.apply_to_environ`` — and are
clamped to their bounds rather than trusted.

``HEADROOM_JEVK5=auto|on|off`` is a typed *service policy*, not a rollout
feature:

* ``auto`` (default) — use the advisor opportunistically; never block wrap or
  proxy startup; first-run background setup only when online.
* ``on`` — same request behavior (requests always fail open), but setup
  failures are surfaced as explicit diagnostics.
* ``off`` — no service starts and no advisor calls.

Endpoints are distinct and never inferred from one another:
``HEADROOM_JEVK5_LLAMA_URL`` is an operator-managed llama-server (``/health``,
``/tokenize``, ``/completion``); ``HEADROOM_JEVK5_SYSTEMONE_URL`` is an
operator-managed JevK5 ``/v1/systemone`` service.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

from headroom import paths

JEVK5_RUNTIME_VERSION = "0.3.3"
JEVK5_PIP_SPEC = "jevk5 @ git+https://github.com/allebee/jevk5@v0.3.3"
DEFAULT_MODEL_REPO = "alibiserikbay/JevK5-GGUF"
DEFAULT_MODEL_FILE = "jevk5-4b-v0.3-Q8_0.gguf"
DEFAULT_MODEL_SHA256 = "aea433883bc7ed399f2fbd539e53d2eac7caf71a946fe6650995a413979d4a30"
DEFAULT_MODEL_BYTES_APPROX = 4_482_402_720
# The llama.cpp revision the published JevK5 GGUF files were made and checked at.
DEFAULT_LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"
DEFAULT_LLAMA_CPP_REF = "9575389"


class ServicePolicy(str, Enum):
    AUTO = "auto"
    ON = "on"
    OFF = "off"

    @classmethod
    def parse(cls, value: str | None) -> ServicePolicy:
        normalized = (value or "").strip().lower()
        if normalized in ("", "auto", "default"):
            return cls.AUTO
        if normalized in ("1", "on", "true", "yes", "enabled"):
            return cls.ON
        if normalized in ("0", "off", "false", "no", "disabled"):
            return cls.OFF
        return cls.AUTO


def _float(env: Mapping[str, str], name: str, default: float, lo: float, hi: float) -> float:
    try:
        value = float(env.get(name, "") or default)
    except ValueError:
        value = default
    return min(hi, max(lo, value))


def _int(env: Mapping[str, str], name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(float(env.get(name, "") or default))
    except ValueError:
        value = default
    return min(hi, max(lo, value))


def _flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = (env.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on", "enabled")


@dataclass(frozen=True)
class IntelligenceSettings:
    policy: ServicePolicy = ServicePolicy.AUTO
    autostart: bool = True
    # Model
    model_repo: str = DEFAULT_MODEL_REPO
    model_file: str = DEFAULT_MODEL_FILE
    model_sha256: str | None = DEFAULT_MODEL_SHA256
    model_path_override: str | None = None
    model_operator_managed: bool = False
    # Calibration (JevK5 v0.3 4B Q8_0 as published for runtime v0.3.3)
    temperature: float = 1.22
    knockout_temperature: float = 0.93
    top_k: int = 40
    missing_retry_n_probs: int = 128
    # llama-server
    llama_server_override: str | None = None
    llama_url: str | None = None
    systemone_url: str | None = None
    host: str = "127.0.0.1"
    preferred_port: int = 8091
    ctx_size: int = 8192
    gpu_layers: str = "auto"
    load_timeout_s: float = 240.0
    llama_cpp_repo: str = DEFAULT_LLAMA_CPP_REPO
    llama_cpp_ref: str = DEFAULT_LLAMA_CPP_REF
    # Advisor hot-path limits
    decision_timeout_ms: int = 1200
    calls_per_request: int = 4
    cache_ttl_s: float = 30.0
    cache_max_entries: int = 512
    min_confidence: float = 0.60
    weight: float = 0.25
    weight_cap: float = 0.40
    breaker_failures: int = 3
    breaker_open_s: float = 60.0
    max_state_tokens: int = 6000
    max_options: int = 256

    @property
    def effective_weight(self) -> float:
        return min(self.weight, self.weight_cap, 0.40)

    @property
    def model_dir(self) -> Path:
        return paths.intelligence_dir() / "models"

    @property
    def model_path(self) -> Path:
        if self.model_path_override:
            return Path(self.model_path_override).expanduser()
        return self.model_dir / self.model_file

    @property
    def custom_model(self) -> bool:
        return bool(self.model_path_override) or self.model_file != DEFAULT_MODEL_FILE

    @property
    def expected_sha256(self) -> str | None:
        """Trusted checksum for the configured model, or ``None`` (operator-managed)."""
        return self.model_sha256

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> IntelligenceSettings:
        env = os.environ if environ is None else environ
        model_file = (env.get("HEADROOM_JEVK5_MODEL_FILE") or DEFAULT_MODEL_FILE).strip()
        model_path = (env.get("HEADROOM_JEVK5_MODEL") or "").strip() or None
        custom = bool(model_path) or model_file != DEFAULT_MODEL_FILE
        custom_sha = (env.get("HEADROOM_JEVK5_MODEL_SHA256") or "").strip().lower() or None
        operator_managed = _flag(env, "HEADROOM_JEVK5_MODEL_OPERATOR_MANAGED", False)
        # The default hash belongs to the default file only: a custom model
        # must bring its own checksum or be explicitly operator-managed.
        sha = custom_sha if custom else DEFAULT_MODEL_SHA256
        weight_cap = _float(env, "HEADROOM_JEVK5_WEIGHT_CAP", 0.40, 0.0, 0.40)
        return cls(
            policy=ServicePolicy.parse(env.get("HEADROOM_JEVK5")),
            autostart=_flag(env, "HEADROOM_JEVK5_AUTOSTART", True),
            model_repo=(env.get("HEADROOM_JEVK5_MODEL_REPO") or DEFAULT_MODEL_REPO).strip(),
            model_file=model_file,
            model_sha256=sha,
            model_path_override=model_path,
            model_operator_managed=operator_managed and custom,
            temperature=_float(env, "HEADROOM_JEVK5_TEMPERATURE", 1.22, 0.05, 10.0),
            knockout_temperature=_float(
                env, "HEADROOM_JEVK5_KNOCKOUT_TEMPERATURE", 0.93, 0.05, 10.0
            ),
            top_k=_int(env, "HEADROOM_JEVK5_TOP_K", 40, 16, 512),
            missing_retry_n_probs=_int(env, "HEADROOM_JEVK5_MISSING_RETRY_N_PROBS", 128, 16, 1024),
            llama_server_override=(env.get("HEADROOM_JEVK5_LLAMA_SERVER") or "").strip() or None,
            llama_url=(env.get("HEADROOM_JEVK5_LLAMA_URL") or "").strip().rstrip("/") or None,
            systemone_url=(env.get("HEADROOM_JEVK5_SYSTEMONE_URL") or "").strip().rstrip("/")
            or None,
            preferred_port=_int(env, "HEADROOM_JEVK5_PORT", 8091, 1024, 65535),
            ctx_size=_int(env, "HEADROOM_JEVK5_CTX_SIZE", 8192, 2048, 16384),
            gpu_layers=(env.get("HEADROOM_JEVK5_GPU_LAYERS") or "auto").strip() or "auto",
            load_timeout_s=_float(env, "HEADROOM_JEVK5_LOAD_TIMEOUT_S", 240.0, 10.0, 1800.0),
            llama_cpp_ref=(env.get("HEADROOM_JEVK5_LLAMA_CPP_REF") or DEFAULT_LLAMA_CPP_REF).strip(),
            decision_timeout_ms=_int(env, "HEADROOM_JEVK5_DECISION_TIMEOUT_MS", 1200, 100, 10000),
            calls_per_request=_int(env, "HEADROOM_JEVK5_CALLS_PER_REQUEST", 4, 0, 16),
            cache_ttl_s=_float(env, "HEADROOM_JEVK5_CACHE_TTL_S", 30.0, 0.0, 600.0),
            min_confidence=_float(env, "HEADROOM_JEVK5_MIN_CONFIDENCE", 0.60, 0.0, 1.0),
            weight=_float(env, "HEADROOM_JEVK5_WEIGHT", 0.25, 0.0, weight_cap),
            weight_cap=weight_cap,
            breaker_failures=_int(env, "HEADROOM_JEVK5_BREAKER_FAILURES", 3, 1, 100),
            breaker_open_s=_float(env, "HEADROOM_JEVK5_BREAKER_OPEN_S", 60.0, 1.0, 3600.0),
            max_state_tokens=_int(env, "HEADROOM_JEVK5_MAX_STATE_TOKENS", 6000, 256, 14000),
        )

    def with_overrides(self, **changes: object) -> IntelligenceSettings:
        return replace(self, **changes)  # type: ignore[arg-type]

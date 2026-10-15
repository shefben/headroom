"""Tool-result admission, structured deltas and family interceptors (unified-21 #4, #5, #14).

Extension surface for later phases (consume, never replace):

* :func:`~headroom.agent.tool_admission.families.build_envelope` — the one
  :class:`~headroom.agent.interfaces.ToolResultEnvelope` builder (family,
  stable ``resource_id``, version hash, explicit partiality);
* :class:`~headroom.agent.tool_admission.dispatcher.ToolAdmissionDispatcher`
  / :func:`get_dispatcher` — the single dispatcher (CCR-first storage
  sequence, delta, family view, SafetyGate) behind both entry points;
* :data:`~headroom.agent.tool_admission.views.INTERCEPTORS` — one
  ``ToolInterceptor`` per family (``view(envelope, ctx) -> FamilyView``);
* :class:`~headroom.agent.tool_admission.delta.ResourceVersionStore` and
  :func:`~headroom.agent.tool_admission.delta.compute_delta`;
* :mod:`~headroom.agent.tool_admission.host` — the Claude Code hook adapter
  and :func:`~headroom.agent.tool_admission.host.host_capability`;
* :class:`~headroom.agent.tool_admission.transform.ToolAdmissionTransform` —
  the proxy live-zone fallback (first pipeline transform).
"""

from headroom.agent.tool_admission.families import (
    INTERCEPTED_FAMILIES,
    ToolFamily,
    build_envelope,
    classify,
)

FEATURE_TOOL_ADMISSION = "agent_tool_admission"

__all__ = [
    "FEATURE_TOOL_ADMISSION",
    "INTERCEPTED_FAMILIES",
    "ToolFamily",
    "build_envelope",
    "classify",
]

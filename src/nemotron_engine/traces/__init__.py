"""Pass 5 trace compiler and anti-imitation audit APIs."""

from .dedupe import TraceDedupeReport, compute_trace_hash, dedupe_traces
from .templates import (
    render_direct_answer_trace,
    render_hard_rejection_trace,
    render_minimal_induced_rule_trace,
    render_minimal_known_rule_trace,
)
from .trace_auditor import TraceAuditError, assert_trace_valid, audit_trace
from .trace_compiler import (
    TraceAuditReport,
    TraceCompilationError,
    TraceRecord,
    TraceType,
    compile_rejection_trace,
    compile_trace_from_proof,
    compile_trace_from_solver_result,
)

__all__ = [
    "TraceAuditError",
    "TraceAuditReport",
    "TraceCompilationError",
    "TraceDedupeReport",
    "TraceRecord",
    "TraceType",
    "assert_trace_valid",
    "audit_trace",
    "compile_rejection_trace",
    "compile_trace_from_proof",
    "compile_trace_from_solver_result",
    "compute_trace_hash",
    "dedupe_traces",
    "render_direct_answer_trace",
    "render_hard_rejection_trace",
    "render_minimal_induced_rule_trace",
    "render_minimal_known_rule_trace",
]

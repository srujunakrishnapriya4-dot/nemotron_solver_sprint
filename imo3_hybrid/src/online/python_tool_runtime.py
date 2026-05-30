from __future__ import annotations

import ast
from dataclasses import dataclass, field
import io
import json
import traceback
from typing import Any, Mapping
import re
from contextlib import redirect_stderr, redirect_stdout
import subprocess
import sys


_PYTHON_TOOL_TAG_RE = re.compile(
    r"<python_tool>\s*(.*?)\s*</python_tool>",
    re.IGNORECASE | re.DOTALL,
)
_PYTHON_TOOL_FENCE_RE = re.compile(
    r"```python-tool\s*(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_ALLOWED_IMPORT_ROOTS = {
    "math",
    "numpy",
    "sympy",
    "itertools",
    "collections",
    "mpmath",
}
_STRUCTURED_TOOL_NAMES = {
    "python",
    "python_tool",
    "python_exec",
    "jupyter",
    "execute_python",
}

_SUBPROCESS_WORKER_BOOTSTRAP = r"""
import json
import sys
import traceback

payload = json.load(sys.stdin)
for entry in payload.get("sys_path", []):
    if entry and entry not in sys.path:
        sys.path.insert(0, entry)

try:
    from src.online.python_tool_runtime import PythonToolConfig, _execute_code_bundle
    config = PythonToolConfig(**dict(payload.get("config", {}) or {}))
    result = _execute_code_bundle(
        list(payload.get("code_history", []) or []),
        str(payload.get("current_code", "") or ""),
        config,
    )
except Exception as exc:
    result = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "result_repr": None,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "timeout": False,
        "execution_time_sec": 0.0,
        "metadata": {"traceback": traceback.format_exc(limit=8)},
    }

sys.stdout.write(json.dumps(result))
"""


@dataclass(frozen=True)
class PythonToolConfig:
    max_turns_per_attempt: int = 8
    max_tool_calls_per_attempt: int = 6
    max_execution_time_sec: float = 5.0
    max_stdout_chars: int = 4000
    max_stderr_chars: int = 2000
    max_code_chars: int = 6000
    mpmath_precision: int = 64
    allow_filesystem: bool = False
    allow_network: bool = False


@dataclass(frozen=True)
class PythonToolRequest:
    code: str
    source: str
    raw_payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PythonToolResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    result_repr: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    timeout: bool = False
    execution_time_sec: float = 0.0
    source: str = "text_protocol_adapter"
    metadata: Mapping[str, Any] = field(default_factory=dict)


def detect_python_tool_request(
    text: str,
    *,
    raw_output: Mapping[str, Any] | None = None,
) -> PythonToolRequest | None:
    structured = _detect_structured_tool_request(raw_output)
    if structured is not None:
        return structured

    payload = str(text or "")
    tagged_matches = list(_PYTHON_TOOL_TAG_RE.finditer(payload))
    if tagged_matches:
        code = tagged_matches[-1].group(1).strip()
        if code:
            return PythonToolRequest(
                code=code,
                source="text_protocol_adapter",
                raw_payload={"protocol": "xml_tag"},
            )

    fenced_matches = list(_PYTHON_TOOL_FENCE_RE.finditer(payload))
    if fenced_matches:
        code = fenced_matches[-1].group(1).strip()
        if code:
            return PythonToolRequest(
                code=code,
                source="text_protocol_adapter",
                raw_payload={"protocol": "python_tool_fence"},
            )
    return None


def render_python_tool_feedback(result: PythonToolResult) -> str:
    status = "ok" if result.success else "error"
    lines = [
        f"<python_tool_result status=\"{status}\" timeout=\"{str(result.timeout).lower()}\" source=\"{result.source}\">",
        f"stdout:\n{result.stdout or '<empty>'}",
        f"stderr:\n{result.stderr or '<empty>'}",
        f"result:\n{result.result_repr or '<none>'}",
        f"error_type:\n{result.error_type or '<none>'}",
        f"error_message:\n{result.error_message or '<none>'}",
        f"execution_time_sec:\n{result.execution_time_sec:.4f}",
        "</python_tool_result>",
    ]
    return "\n".join(lines)


class PythonToolSession:
    def __init__(
        self,
        *,
        attempt_id: str,
        config: PythonToolConfig | None = None,
    ) -> None:
        self.attempt_id = str(attempt_id)
        self.config = config or PythonToolConfig()
        self._history: list[str] = []
        self.tool_call_count = 0
        self.tool_error_count = 0

    @property
    def history(self) -> tuple[str, ...]:
        return tuple(self._history)

    def execute(self, request: PythonToolRequest) -> PythonToolResult:
        code = str(request.code or "").strip()
        if not code:
            self.tool_error_count += 1
            return PythonToolResult(
                success=False,
                error_type="EmptyToolCode",
                error_message="Python tool request contained no executable code.",
                source=request.source,
            )
        if len(code) > self.config.max_code_chars:
            self.tool_error_count += 1
            return PythonToolResult(
                success=False,
                error_type="ToolCodeTooLong",
                error_message=(
                    f"Python tool request exceeded max_code_chars={self.config.max_code_chars}."
                ),
                source=request.source,
            )
        if self.tool_call_count >= self.config.max_tool_calls_per_attempt:
            self.tool_error_count += 1
            return PythonToolResult(
                success=False,
                error_type="ToolCallLimitReached",
                error_message="Per-attempt Python tool-call limit reached.",
                source=request.source,
            )

        self.tool_call_count += 1
        result = _execute_in_subprocess(
            code_history=self._history,
            current_code=code,
            config=self.config,
            source=request.source,
        )
        if result.success:
            self._history.append(code)
        else:
            self.tool_error_count += 1
        return result


def _detect_structured_tool_request(raw_output: Mapping[str, Any] | None) -> PythonToolRequest | None:
    if not isinstance(raw_output, Mapping):
        return None
    tool_calls = raw_output.get("tool_calls")
    if not tool_calls:
        return None
    calls = tool_calls if isinstance(tool_calls, (list, tuple)) else [tool_calls]
    for call in calls:
        if not isinstance(call, Mapping):
            continue
        name = str(call.get("name", "") or call.get("tool_name", "")).strip().lower()
        if name not in _STRUCTURED_TOOL_NAMES:
            continue
        code = _extract_structured_code(call.get("arguments"))
        if code:
            return PythonToolRequest(
                code=code,
                source="structured_tool_calls",
                raw_payload=dict(call),
            )
    return None


def _extract_structured_code(arguments: Any) -> str | None:
    if arguments is None:
        return None
    if isinstance(arguments, str):
        return arguments.strip() or None
    if isinstance(arguments, Mapping):
        for key in ("code", "input", "python", "content"):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _execute_in_subprocess(
    *,
    code_history: list[str],
    current_code: str,
    config: PythonToolConfig,
    source: str,
) -> PythonToolResult:
    payload = {
        "code_history": list(code_history),
        "current_code": current_code,
        "config": {
            "max_turns_per_attempt": config.max_turns_per_attempt,
            "max_tool_calls_per_attempt": config.max_tool_calls_per_attempt,
            "max_execution_time_sec": config.max_execution_time_sec,
            "max_stdout_chars": config.max_stdout_chars,
            "max_stderr_chars": config.max_stderr_chars,
            "max_code_chars": config.max_code_chars,
            "mpmath_precision": config.mpmath_precision,
            "allow_filesystem": config.allow_filesystem,
            "allow_network": config.allow_network,
        },
        "sys_path": list(sys.path),
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_WORKER_BOOTSTRAP],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=float(config.max_execution_time_sec),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return PythonToolResult(
            success=False,
            error_type="ToolTimeout",
            error_message=(
                f"Python tool call exceeded max_execution_time_sec={config.max_execution_time_sec}."
            ),
            timeout=True,
            source=source,
        )
    if completed.returncode != 0 and not completed.stdout.strip():
        return PythonToolResult(
            success=False,
            error_type="ToolWorkerCrashed",
            error_message=f"Python tool worker exited with code {completed.returncode}.",
            source=source,
            metadata={"stderr": completed.stderr[-config.max_stderr_chars :]},
        )
    try:
        payload = json.loads(completed.stdout or "{}")
    except Exception:
        return PythonToolResult(
            success=False,
            error_type="ToolNoResult",
            error_message="Python tool worker produced no parseable result payload.",
            source=source,
            metadata={
                "stdout": (completed.stdout or "")[-config.max_stdout_chars :],
                "stderr": (completed.stderr or "")[-config.max_stderr_chars :],
            },
        )
    return PythonToolResult(
        success=bool(payload.get("success", False)),
        stdout=str(payload.get("stdout", "") or ""),
        stderr=str(payload.get("stderr", "") or ""),
        result_repr=payload.get("result_repr"),
        error_type=payload.get("error_type"),
        error_message=payload.get("error_message"),
        timeout=bool(payload.get("timeout", False)),
        execution_time_sec=float(payload.get("execution_time_sec", 0.0) or 0.0),
        source=source,
        metadata=dict(payload.get("metadata", {}) or {}),
    )


def _execute_code_bundle(
    code_history: list[str],
    current_code: str,
    config: PythonToolConfig,
) -> dict[str, Any]:
    import collections
    import importlib
    import itertools
    import math
    import time

    optional_modules: dict[str, Any] = {}
    unavailable_modules: dict[str, str] = {}
    for module_name in ("numpy", "sympy", "mpmath"):
        try:
            optional_modules[module_name] = importlib.import_module(module_name)
        except Exception as exc:
            unavailable_modules[module_name] = f"{type(exc).__name__}: {exc}"

    mpmath = optional_modules.get("mpmath")
    if mpmath is not None:
        mpmath.mp.dps = int(config.mpmath_precision)
    started = time.perf_counter()
    allowed_import_roots = set(_ALLOWED_IMPORT_ROOTS)
    if config.allow_filesystem:
        allowed_import_roots.update({"os", "pathlib", "io"})
    if config.allow_network:
        allowed_import_roots.update({"socket", "ssl", "urllib", "http"})

    def _limited_import(name: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (), level: int = 0) -> Any:
        del globals_, locals_, fromlist, level
        root = str(name or "").split(".", 1)[0]
        if root not in allowed_import_roots:
            raise ImportError(f"Import '{name}' is not allowed in the bounded Python tool runtime.")
        return importlib.import_module(name)

    allowed_builtins = {
        "__import__": _limited_import,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "RuntimeError": RuntimeError,
        "AssertionError": AssertionError,
        "ArithmeticError": ArithmeticError,
        "ZeroDivisionError": ZeroDivisionError,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "pow": pow,
        "print": print,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    if config.allow_filesystem:
        allowed_builtins["open"] = open
    globals_dict: dict[str, Any] = {
        "__builtins__": allowed_builtins,
        "__name__": "__python_tool_runtime__",
        "math": math,
        "itertools": itertools,
        "collections": collections,
    }
    if "numpy" in optional_modules:
        globals_dict["numpy"] = optional_modules["numpy"]
        globals_dict["np"] = optional_modules["numpy"]
    if "sympy" in optional_modules:
        globals_dict["sympy"] = optional_modules["sympy"]
        globals_dict["sp"] = optional_modules["sympy"]
    if mpmath is not None:
        globals_dict["mpmath"] = mpmath
        globals_dict["mp"] = mpmath

    silent_stdout = io.StringIO()
    silent_stderr = io.StringIO()
    for prior_code in code_history:
        _execute_single_cell(
            prior_code,
            globals_dict,
            stdout_buffer=silent_stdout,
            stderr_buffer=silent_stderr,
        )

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        result_repr = _execute_single_cell(
            current_code,
            globals_dict,
            stdout_buffer=stdout_buffer,
            stderr_buffer=stderr_buffer,
        )
        success = True
        error_type = None
        error_message = None
    except Exception as exc:
        success = False
        result_repr = None
        error_type = type(exc).__name__
        error_message = str(exc)
        stderr_buffer.write(traceback.format_exc(limit=5))

    return {
        "success": success,
        "stdout": stdout_buffer.getvalue()[: int(config.max_stdout_chars)],
        "stderr": stderr_buffer.getvalue()[: int(config.max_stderr_chars)],
        "result_repr": result_repr,
        "error_type": error_type,
        "error_message": error_message,
        "timeout": False,
        "execution_time_sec": round(time.perf_counter() - started, 6),
        "metadata": {
            "history_cell_count": len(code_history),
            "unavailable_modules": unavailable_modules,
        },
    }


def _execute_single_cell(
    code: str,
    globals_dict: dict[str, Any],
    *,
    stdout_buffer: io.StringIO,
    stderr_buffer: io.StringIO,
) -> str | None:
    parsed = ast.parse(code, mode="exec")
    final_expr: ast.expr | None = None
    body = list(parsed.body)
    if body and isinstance(body[-1], ast.Expr):
        final_expr = body.pop().value
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        if body:
            exec(compile(module, "<python_tool>", "exec"), globals_dict, globals_dict)
        if final_expr is None:
            return None
        value = eval(compile(ast.Expression(final_expr), "<python_tool>", "eval"), globals_dict, globals_dict)
    return repr(value)


__all__ = [
    "PythonToolConfig",
    "PythonToolRequest",
    "PythonToolResult",
    "PythonToolSession",
    "detect_python_tool_request",
    "render_python_tool_feedback",
]

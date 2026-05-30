"""Structured logging, tracing, and runtime diagnostics for the repo.

This module is intentionally architecture-aware:
- structured event records instead of ad-hoc string formatting
- explicit stage/component metadata for parsing, routing, search, verifier, etc.
- deterministic JSON/text formatting for scripts and tests
- lightweight, standard-library-first implementation with optional loguru bridge
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import date, datetime, time, timezone
from enum import Enum
import json
import logging as py_logging
from pathlib import Path
import sys
import traceback
from typing import Any, Callable, Mapping, Sequence, TextIO

from src.common.utils import stable_sort_key


TRACE_LEVEL_NUM = 5
if py_logging.getLevelName(TRACE_LEVEL_NUM) == f"Level {TRACE_LEVEL_NUM}":
    py_logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")


def _trace_logger(self: py_logging.Logger, message: str, *args: Any, **kwargs: Any) -> None:
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kwargs)


if not hasattr(py_logging.Logger, "trace"):
    py_logging.Logger.trace = _trace_logger  # type: ignore[attr-defined]


class LogLevel(str, Enum):
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

    @property
    def numeric(self) -> int:
        return {
            LogLevel.TRACE: TRACE_LEVEL_NUM,
            LogLevel.DEBUG: py_logging.DEBUG,
            LogLevel.INFO: py_logging.INFO,
            LogLevel.WARNING: py_logging.WARNING,
            LogLevel.ERROR: py_logging.ERROR,
        }[self]


class EventStage(str, Enum):
    GENERAL = "general"
    PARSING = "parsing"
    ROUTING = "routing"
    BRANCH_SEARCH = "branch_search"
    SYMBOLIC_VALIDATION = "symbolic_validation"
    VERIFIER_EVALUATION = "verifier_evaluation"
    AGGREGATION = "aggregation"
    OFFLINE_ARTIFACT = "offline_artifact"
    ONLINE_RUNTIME = "online_runtime"
    SERVING = "serving"


@dataclass(frozen=True)
class VerbosityControl:
    max_string_length: int = 4000
    max_collection_items: int = 32
    max_mapping_items: int = 48
    max_depth: int = 6
    max_traceback_lines: int = 24


@dataclass(frozen=True)
class LoggingContext:
    run_id: str | None = None
    problem_id: str | None = None
    branch_id: str | None = None
    artifact_id: str | None = None
    stage: str | None = None
    module: str | None = None
    component: str | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def bind(self, **updates: Any) -> "LoggingContext":
        merged_tags = tuple(self.tags)
        if "tags" in updates and updates["tags"] is not None:
            merged_tags = _normalize_tags(tuple(merged_tags) + tuple(updates.pop("tags")))

        metadata_update = updates.pop("metadata", None)
        merged_metadata = dict(self.metadata)
        if isinstance(metadata_update, Mapping):
            merged_metadata.update(dict(metadata_update))

        staged = replace(self, tags=merged_tags, metadata=merged_metadata)
        valid_fields = set(staged.__dataclass_fields__.keys())
        direct_updates = {key: value for key, value in updates.items() if key in valid_fields and value is not None}
        if direct_updates:
            staged = replace(staged, **direct_updates)
        unknown = {key: value for key, value in updates.items() if key not in valid_fields and value is not None}
        if unknown:
            new_meta = dict(staged.metadata)
            new_meta.update(unknown)
            staged = replace(staged, metadata=new_meta)
        return staged

    def as_dict(self, serializer: "JsonSafeSerializer") -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key in ("run_id", "problem_id", "branch_id", "artifact_id", "stage", "module", "component"):
            value = getattr(self, key)
            if value:
                payload[key] = value
        if self.tags:
            payload["tags"] = list(self.tags)
        if self.metadata:
            payload["metadata"] = serializer.convert(self.metadata)
        return payload


@dataclass(frozen=True)
class ExceptionSnapshot:
    error_type: str
    error_module: str
    message: str
    args: tuple[Any, ...] = ()
    traceback_lines: tuple[str, ...] = ()

    def as_dict(self, serializer: "JsonSafeSerializer") -> dict[str, Any]:
        payload = {
            "type": self.error_type,
            "module": self.error_module,
            "message": self.message,
        }
        if self.args:
            payload["args"] = serializer.convert(list(self.args))
        if self.traceback_lines:
            payload["traceback"] = list(self.traceback_lines)
        return payload


@dataclass(frozen=True)
class StructuredEvent:
    timestamp: str
    logger_name: str
    level: str
    event: str
    message: str
    context: LoggingContext = field(default_factory=LoggingContext)
    payload: Mapping[str, Any] = field(default_factory=dict)
    error: ExceptionSnapshot | None = None

    def as_dict(self, serializer: "JsonSafeSerializer") -> dict[str, Any]:
        record = {
            "timestamp": self.timestamp,
            "logger": self.logger_name,
            "level": self.level,
            "event": self.event,
            "message": self.message,
            "context": self.context.as_dict(serializer),
            "payload": serializer.convert(self.payload),
        }
        if self.error is not None:
            record["error"] = self.error.as_dict(serializer)
        return record


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    created_at: str
    experiment_name: str = ""
    tags: tuple[str, ...] = ()
    config: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self, serializer: "JsonSafeSerializer") -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "experiment_name": self.experiment_name,
            "tags": list(self.tags),
            "config": serializer.convert(self.config),
            "metadata": serializer.convert(self.metadata),
        }


@dataclass(frozen=True)
class ArtifactManifest:
    artifact_id: str
    created_at: str
    artifact_type: str
    path: str | None = None
    run_id: str | None = None
    problem_id: str | None = None
    stage: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self, serializer: "JsonSafeSerializer") -> dict[str, Any]:
        payload = {
            "artifact_id": self.artifact_id,
            "created_at": self.created_at,
            "artifact_type": self.artifact_type,
            "metadata": serializer.convert(self.metadata),
        }
        if self.path:
            payload["path"] = self.path
        if self.run_id:
            payload["run_id"] = self.run_id
        if self.problem_id:
            payload["problem_id"] = self.problem_id
        if self.stage:
            payload["stage"] = self.stage
        return payload


@dataclass(frozen=True)
class RepoLoggingConfig:
    logger_name: str = "imo3_hybrid"
    level: LogLevel | str = LogLevel.INFO
    json_output: bool = True
    include_source_location: bool = False
    propagate: bool = False
    apply_to_root: bool = False
    reset_handlers: bool = True
    capture_warnings: bool = False
    bridge_loguru: bool = False
    stream: TextIO | None = None
    log_file: str | None = None
    verbosity: VerbosityControl = field(default_factory=VerbosityControl)

    def normalized_level(self) -> LogLevel:
        return _coerce_level(self.level)


class JsonSafeSerializer:
    """Deterministic, bounded serializer for structured log payloads."""

    def __init__(self, verbosity: VerbosityControl | None = None) -> None:
        self.verbosity = verbosity or VerbosityControl()

    def convert(self, value: Any, *, depth: int = 0) -> Any:
        if depth >= self.verbosity.max_depth:
            return self._truncate_string(repr(value))
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            if value != value:
                return "nan"
            if value == float("inf"):
                return "inf"
            if value == float("-inf"):
                return "-inf"
            return value
        if isinstance(value, str):
            return self._truncate_string(value)
        if isinstance(value, (datetime, date, time)):
            return _isoformat(value)
        if isinstance(value, Enum):
            return self.convert(value.value, depth=depth + 1)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, BaseException):
            return capture_exception(value, verbosity=self.verbosity).as_dict(self)
        if is_dataclass(value):
            return self.convert(asdict(value), depth=depth + 1)
        if hasattr(value, "model_dump") and callable(value.model_dump):
            try:
                return self.convert(value.model_dump(), depth=depth + 1)
            except Exception:
                return self._truncate_string(repr(value))
        if hasattr(value, "dict") and callable(value.dict):
            try:
                return self.convert(value.dict(), depth=depth + 1)
            except Exception:
                return self._truncate_string(repr(value))
        if hasattr(value, "tolist") and callable(value.tolist):
            try:
                return self.convert(value.tolist(), depth=depth + 1)
            except Exception:
                return self._truncate_string(repr(value))
        if isinstance(value, Mapping):
            items = sorted((str(key), val) for key, val in value.items())
            if len(items) > self.verbosity.max_mapping_items:
                kept = items[: self.verbosity.max_mapping_items]
                suffix = [("_truncated_items", len(items) - len(kept))]
                items = kept + suffix
            return {key: self.convert(val, depth=depth + 1) for key, val in items}
        if isinstance(value, (set, frozenset)):
            seq = sorted(value, key=stable_sort_key)
            if len(seq) > self.verbosity.max_collection_items:
                seq = seq[: self.verbosity.max_collection_items] + [
                    f"...({len(value) - self.verbosity.max_collection_items} more)"
                ]
            return [self.convert(item, depth=depth + 1) for item in seq]
        if isinstance(value, (list, tuple)):
            seq = list(value)
            if len(seq) > self.verbosity.max_collection_items:
                seq = seq[: self.verbosity.max_collection_items] + [
                    f"...({len(value) - self.verbosity.max_collection_items} more)"
                ]
            return [self.convert(item, depth=depth + 1) for item in seq]
        if isinstance(value, bytes):
            return self._truncate_string(value.decode("utf-8", errors="replace"))
        if hasattr(value, "__dict__"):
            public = {
                str(key): val
                for key, val in vars(value).items()
                if not str(key).startswith("_")
            }
            if public:
                return self.convert(public, depth=depth + 1)
        return self._truncate_string(repr(value))

    def dumps(self, value: Any) -> str:
        return json.dumps(
            self.convert(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def _truncate_string(self, value: str) -> str:
        if len(value) <= self.verbosity.max_string_length:
            return value
        clipped = value[: self.verbosity.max_string_length]
        return f"{clipped}...({len(value) - self.verbosity.max_string_length} more chars)"


class StructuredLogFormatter(py_logging.Formatter):
    def __init__(
        self,
        *,
        json_output: bool = True,
        include_source_location: bool = False,
        verbosity: VerbosityControl | None = None,
    ) -> None:
        super().__init__()
        self.json_output = json_output
        self.include_source_location = include_source_location
        self.serializer = JsonSafeSerializer(verbosity=verbosity)

    def format(self, record: py_logging.LogRecord) -> str:
        data = getattr(record, "structured_event_dict", None)
        if not isinstance(data, Mapping):
            data = self._fallback_record(record)
        if self.json_output:
            return self.serializer.dumps(data)
        return self._format_text(data)

    def _fallback_record(self, record: py_logging.LogRecord) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.include_source_location:
            payload["source"] = {
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
                "path": record.pathname,
            }
        if record.exc_info:
            exc = record.exc_info[1]
            if isinstance(exc, BaseException):
                payload["error"] = capture_exception(exc, verbosity=self.serializer.verbosity).as_dict(self.serializer)
        return {
            "timestamp": _isoformat(datetime.fromtimestamp(record.created, tz=timezone.utc)),
            "logger": record.name,
            "level": record.levelname,
            "event": getattr(record, "event_name", "unstructured_log"),
            "message": record.getMessage(),
            "context": {},
            "payload": payload,
        }

    def _format_text(self, data: Mapping[str, Any]) -> str:
        timestamp = str(data.get("timestamp", ""))
        level = str(data.get("level", "INFO"))
        logger_name = str(data.get("logger", "imo3_hybrid"))
        event = str(data.get("event", "log"))
        message = str(data.get("message", ""))
        context = data.get("context", {}) or {}
        payload = data.get("payload", {}) or {}
        error = data.get("error")

        context_parts: list[str] = []
        for key in ("run_id", "problem_id", "branch_id", "artifact_id", "stage", "module", "component"):
            value = context.get(key)
            if value:
                context_parts.append(f"{key}={value}")
        if context.get("tags"):
            context_parts.append(f"tags={','.join(str(item) for item in context['tags'])}")

        segments = [timestamp, level, logger_name, event]
        if context_parts:
            segments.append(" ".join(context_parts))
        if message:
            segments.append(message)
        if payload:
            segments.append(f"payload={self.serializer.dumps(payload)}")
        if error:
            segments.append(f"error={self.serializer.dumps(error)}")
        return " | ".join(segment for segment in segments if segment)


class StructuredLogger:
    """Context-bound structured logger with stage-aware helpers."""

    def __init__(
        self,
        logger: py_logging.Logger,
        *,
        context: LoggingContext | Mapping[str, Any] | None = None,
        serializer: JsonSafeSerializer | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._logger = logger
        self._context = _coerce_context(context)
        self._serializer = serializer or JsonSafeSerializer()
        self._clock = clock or _utc_now

    @property
    def name(self) -> str:
        return self._logger.name

    @property
    def context(self) -> LoggingContext:
        return self._context

    @property
    def raw_logger(self) -> py_logging.Logger:
        return self._logger

    def bind(self, **context: Any) -> "StructuredLogger":
        return StructuredLogger(
            self._logger,
            context=self._context.bind(**context),
            serializer=self._serializer,
            clock=self._clock,
        )

    def emit(
        self,
        event: str,
        *,
        level: LogLevel | str = LogLevel.INFO,
        message: str = "",
        payload: Mapping[str, Any] | None = None,
        context: LoggingContext | Mapping[str, Any] | None = None,
        error: BaseException | ExceptionSnapshot | None = None,
    ) -> dict[str, Any]:
        resolved_level = _coerce_level(level)
        merged_context = self._context
        if context is not None:
            merged_context = merged_context.bind(**_context_to_kwargs(context))
        event_record = StructuredEvent(
            timestamp=_isoformat(self._clock()),
            logger_name=self._logger.name,
            level=resolved_level.value,
            event=str(event or "log_event"),
            message=message or str(event or ""),
            context=merged_context,
            payload=dict(payload or {}),
            error=_coerce_error(error, verbosity=self._serializer.verbosity),
        )
        rendered = event_record.as_dict(self._serializer)
        self._logger.log(
            resolved_level.numeric,
            message or str(event or ""),
            extra={
                "structured_event_dict": rendered,
                "event_name": event_record.event,
            },
        )
        return rendered

    def trace(
        self,
        event: str,
        *,
        message: str = "",
        payload: Mapping[str, Any] | None = None,
        context: LoggingContext | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.emit(event, level=LogLevel.TRACE, message=message, payload=payload, context=context)

    def debug(
        self,
        event: str,
        *,
        message: str = "",
        payload: Mapping[str, Any] | None = None,
        context: LoggingContext | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.emit(event, level=LogLevel.DEBUG, message=message, payload=payload, context=context)

    def info(
        self,
        event: str,
        *,
        message: str = "",
        payload: Mapping[str, Any] | None = None,
        context: LoggingContext | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.emit(event, level=LogLevel.INFO, message=message, payload=payload, context=context)

    def warning(
        self,
        event: str,
        *,
        message: str = "",
        payload: Mapping[str, Any] | None = None,
        context: LoggingContext | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.emit(event, level=LogLevel.WARNING, message=message, payload=payload, context=context)

    def error(
        self,
        event: str,
        *,
        message: str = "",
        payload: Mapping[str, Any] | None = None,
        context: LoggingContext | Mapping[str, Any] | None = None,
        error: BaseException | ExceptionSnapshot | None = None,
    ) -> dict[str, Any]:
        return self.emit(
            event,
            level=LogLevel.ERROR,
            message=message,
            payload=payload,
            context=context,
            error=error,
        )

    def exception(
        self,
        event: str,
        *,
        message: str = "",
        payload: Mapping[str, Any] | None = None,
        context: LoggingContext | Mapping[str, Any] | None = None,
        exc: BaseException | None = None,
    ) -> dict[str, Any]:
        resolved_exc = exc or sys.exc_info()[1]
        return self.emit(
            event,
            level=LogLevel.ERROR,
            message=message,
            payload=payload,
            context=context,
            error=resolved_exc,
        )

    def stage_event(
        self,
        stage: EventStage | str,
        event: str,
        *,
        level: LogLevel | str = LogLevel.INFO,
        message: str = "",
        payload: Mapping[str, Any] | None = None,
        context: LoggingContext | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        merged = _coerce_context(context).bind(stage=_coerce_stage(stage).value)
        return self.emit(event, level=level, message=message, payload=payload, context=merged)

    def parsing(self, event: str, **kwargs: Any) -> dict[str, Any]:
        return self.stage_event(EventStage.PARSING, event, **kwargs)

    def routing(self, event: str, **kwargs: Any) -> dict[str, Any]:
        return self.stage_event(EventStage.ROUTING, event, **kwargs)

    def branch_search(self, event: str, **kwargs: Any) -> dict[str, Any]:
        return self.stage_event(EventStage.BRANCH_SEARCH, event, **kwargs)

    def symbolic_validation(self, event: str, **kwargs: Any) -> dict[str, Any]:
        return self.stage_event(EventStage.SYMBOLIC_VALIDATION, event, **kwargs)

    def verifier_evaluation(self, event: str, **kwargs: Any) -> dict[str, Any]:
        return self.stage_event(EventStage.VERIFIER_EVALUATION, event, **kwargs)

    def aggregation(self, event: str, **kwargs: Any) -> dict[str, Any]:
        return self.stage_event(EventStage.AGGREGATION, event, **kwargs)

    def offline_artifact(self, event: str, **kwargs: Any) -> dict[str, Any]:
        return self.stage_event(EventStage.OFFLINE_ARTIFACT, event, **kwargs)

    def online_runtime(self, event: str, **kwargs: Any) -> dict[str, Any]:
        return self.stage_event(EventStage.ONLINE_RUNTIME, event, **kwargs)

    def serving(self, event: str, **kwargs: Any) -> dict[str, Any]:
        return self.stage_event(EventStage.SERVING, event, **kwargs)

    def emit_run_manifest(
        self,
        manifest: RunManifest,
        *,
        level: LogLevel | str = LogLevel.INFO,
        event: str = "run_manifest",
    ) -> dict[str, Any]:
        return self.stage_event(
            EventStage.OFFLINE_ARTIFACT,
            event,
            level=level,
            message=f"Run manifest recorded for {manifest.run_id}",
            payload=manifest.as_dict(self._serializer),
            context=LoggingContext(run_id=manifest.run_id),
        )

    def emit_artifact_manifest(
        self,
        manifest: ArtifactManifest,
        *,
        level: LogLevel | str = LogLevel.INFO,
        event: str = "artifact_manifest",
    ) -> dict[str, Any]:
        return self.stage_event(
            manifest.stage or EventStage.OFFLINE_ARTIFACT.value,
            event,
            level=level,
            message=f"Artifact manifest recorded for {manifest.artifact_id}",
            payload=manifest.as_dict(self._serializer),
            context=LoggingContext(
                run_id=manifest.run_id,
                problem_id=manifest.problem_id,
                artifact_id=manifest.artifact_id,
                stage=manifest.stage,
            ),
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._logger, name)


def configure_repo_logging(config: RepoLoggingConfig | None = None) -> py_logging.Logger:
    """Configure the repo logger deterministically and return the base logger."""

    resolved = config or RepoLoggingConfig()
    target = py_logging.getLogger() if resolved.apply_to_root else py_logging.getLogger(resolved.logger_name)
    target.setLevel(resolved.normalized_level().numeric)
    target.propagate = resolved.propagate

    if resolved.reset_handlers:
        for handler in list(target.handlers):
            target.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    formatter = StructuredLogFormatter(
        json_output=resolved.json_output,
        include_source_location=resolved.include_source_location,
        verbosity=resolved.verbosity,
    )

    handlers: list[py_logging.Handler] = []
    stream_handler = py_logging.StreamHandler(resolved.stream or sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(resolved.normalized_level().numeric)
    handlers.append(stream_handler)

    if resolved.log_file:
        file_handler = py_logging.FileHandler(resolved.log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(resolved.normalized_level().numeric)
        handlers.append(file_handler)

    for handler in handlers:
        target.addHandler(handler)

    if resolved.capture_warnings:
        py_logging.captureWarnings(True)
    if resolved.bridge_loguru:
        bridge_loguru_to_stdlib(
            logger_name=resolved.logger_name,
            level=resolved.normalized_level(),
            verbosity=resolved.verbosity,
        )
    return target


def reset_repo_logging(logger_name: str = "imo3_hybrid", *, apply_to_root: bool = False) -> None:
    target = py_logging.getLogger() if apply_to_root else py_logging.getLogger(logger_name)
    for handler in list(target.handlers):
        target.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def ensure_repo_logging(config: RepoLoggingConfig | None = None) -> py_logging.Logger:
    resolved = config or RepoLoggingConfig()
    target = py_logging.getLogger() if resolved.apply_to_root else py_logging.getLogger(resolved.logger_name)
    if target.handlers:
        return target
    return configure_repo_logging(resolved)


def get_logger(
    name: str = "imo3_hybrid",
    *,
    module: str | None = None,
    component: str | None = None,
    context: LoggingContext | Mapping[str, Any] | None = None,
    config: RepoLoggingConfig | None = None,
    serializer: JsonSafeSerializer | None = None,
) -> StructuredLogger:
    resolved_config = config or RepoLoggingConfig(logger_name=name)
    ensure_repo_logging(resolved_config)
    base_context = _coerce_context(context).bind(module=module, component=component)
    return StructuredLogger(
        py_logging.getLogger(name),
        context=base_context,
        serializer=serializer or JsonSafeSerializer(verbosity=resolved_config.verbosity),
    )


def capture_exception(
    exc: BaseException,
    *,
    verbosity: VerbosityControl | None = None,
) -> ExceptionSnapshot:
    limits = verbosity or VerbosityControl()
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    if len(tb_lines) > limits.max_traceback_lines:
        tb_lines = tb_lines[: limits.max_traceback_lines] + [
            f"...({len(tb_lines) - limits.max_traceback_lines} more traceback lines)\n"
        ]
    return ExceptionSnapshot(
        error_type=exc.__class__.__name__,
        error_module=exc.__class__.__module__,
        message=str(exc),
        args=tuple(getattr(exc, "args", ()) or ()),
        traceback_lines=tuple(line.rstrip("\n") for line in tb_lines),
    )


def make_run_manifest(
    *,
    run_id: str,
    experiment_name: str = "",
    tags: Sequence[str] | None = None,
    config: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    created_at: datetime | None = None,
) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        created_at=_isoformat(created_at or _utc_now()),
        experiment_name=experiment_name,
        tags=_normalize_tags(tags or ()),
        config=dict(config or {}),
        metadata=dict(metadata or {}),
    )


def make_artifact_manifest(
    *,
    artifact_id: str,
    artifact_type: str,
    path: str | Path | None = None,
    run_id: str | None = None,
    problem_id: str | None = None,
    stage: EventStage | str | None = None,
    metadata: Mapping[str, Any] | None = None,
    created_at: datetime | None = None,
) -> ArtifactManifest:
    return ArtifactManifest(
        artifact_id=artifact_id,
        created_at=_isoformat(created_at or _utc_now()),
        artifact_type=artifact_type,
        path=str(path) if path is not None else None,
        run_id=run_id,
        problem_id=problem_id,
        stage=_coerce_stage(stage).value if stage is not None else None,
        metadata=dict(metadata or {}),
    )


def log_parsing_event(logger: StructuredLogger, event: str, **kwargs: Any) -> dict[str, Any]:
    return logger.parsing(event, **kwargs)


def log_routing_event(logger: StructuredLogger, event: str, **kwargs: Any) -> dict[str, Any]:
    return logger.routing(event, **kwargs)


def log_branch_search_event(logger: StructuredLogger, event: str, **kwargs: Any) -> dict[str, Any]:
    return logger.branch_search(event, **kwargs)


def log_symbolic_event(logger: StructuredLogger, event: str, **kwargs: Any) -> dict[str, Any]:
    return logger.symbolic_validation(event, **kwargs)


def log_verifier_event(logger: StructuredLogger, event: str, **kwargs: Any) -> dict[str, Any]:
    return logger.verifier_evaluation(event, **kwargs)


def log_aggregation_event(logger: StructuredLogger, event: str, **kwargs: Any) -> dict[str, Any]:
    return logger.aggregation(event, **kwargs)


def log_offline_artifact_event(logger: StructuredLogger, event: str, **kwargs: Any) -> dict[str, Any]:
    return logger.offline_artifact(event, **kwargs)


def log_online_runtime_event(logger: StructuredLogger, event: str, **kwargs: Any) -> dict[str, Any]:
    return logger.online_runtime(event, **kwargs)


def log_serving_event(logger: StructuredLogger, event: str, **kwargs: Any) -> dict[str, Any]:
    return logger.serving(event, **kwargs)


def bridge_loguru_to_stdlib(
    *,
    logger_name: str = "imo3_hybrid",
    level: LogLevel | str = LogLevel.INFO,
    verbosity: VerbosityControl | None = None,
) -> bool:
    """Optionally mirror loguru events into stdlib logging."""

    try:
        from loguru import logger as loguru_logger  # type: ignore
    except Exception:
        return False

    serializer = JsonSafeSerializer(verbosity=verbosity)
    level_name = _coerce_level(level).value

    def _sink(message: Any) -> None:
        record = message.record
        target_name = str(record.get("name") or logger_name)
        target = py_logging.getLogger(target_name)
        payload = {
            "timestamp": _isoformat(record["time"].astimezone(timezone.utc)),
            "logger": target_name,
            "level": str(record["level"].name),
            "event": "loguru_event",
            "message": str(record["message"]),
            "context": {},
            "payload": serializer.convert(record.get("extra", {})),
        }
        target.log(
            _coerce_level(str(record["level"].name)).numeric,
            str(record["message"]),
            extra={"structured_event_dict": payload, "event_name": "loguru_event"},
        )

    loguru_logger.add(_sink, level=level_name, enqueue=False, backtrace=False, diagnose=False)
    return True


def _coerce_level(level: LogLevel | str) -> LogLevel:
    if isinstance(level, LogLevel):
        return level
    text = str(level or LogLevel.INFO.value).upper()
    if text in {"CRITICAL", "FATAL"}:
        text = "ERROR"
    if text in {"SUCCESS", "NOTICE"}:
        text = "INFO"
    if text == "WARN":
        text = "WARNING"
    return LogLevel[text]


def _coerce_stage(stage: EventStage | str | None) -> EventStage:
    if isinstance(stage, EventStage):
        return stage
    if stage is None:
        return EventStage.GENERAL
    normalized = str(stage).strip().lower()
    for candidate in EventStage:
        if candidate.value == normalized:
            return candidate
    return EventStage.GENERAL


def _coerce_context(context: LoggingContext | Mapping[str, Any] | None) -> LoggingContext:
    if context is None:
        return LoggingContext()
    if isinstance(context, LoggingContext):
        return context
    if not isinstance(context, Mapping):
        return LoggingContext(metadata={"context_repr": repr(context)})
    known = {
        key: context.get(key)
        for key in ("run_id", "problem_id", "branch_id", "artifact_id", "stage", "module", "component")
    }
    tags = _normalize_tags(context.get("tags") or ())
    metadata = context.get("metadata")
    extra = {
        str(key): value
        for key, value in context.items()
        if key not in {"run_id", "problem_id", "branch_id", "artifact_id", "stage", "module", "component", "tags", "metadata"}
    }
    merged_meta = {}
    if isinstance(metadata, Mapping):
        merged_meta.update(dict(metadata))
    merged_meta.update(extra)
    return LoggingContext(
        run_id=_optional_string(known["run_id"]),
        problem_id=_optional_string(known["problem_id"]),
        branch_id=_optional_string(known["branch_id"]),
        artifact_id=_optional_string(known["artifact_id"]),
        stage=_optional_string(known["stage"]),
        module=_optional_string(known["module"]),
        component=_optional_string(known["component"]),
        tags=tags,
        metadata=merged_meta,
    )


def _context_to_kwargs(context: LoggingContext | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(context, LoggingContext):
        data = {
            "run_id": context.run_id,
            "problem_id": context.problem_id,
            "branch_id": context.branch_id,
            "artifact_id": context.artifact_id,
            "stage": context.stage,
            "module": context.module,
            "component": context.component,
            "tags": context.tags,
            "metadata": dict(context.metadata),
        }
        return {key: value for key, value in data.items() if value not in (None, (), {}, "")}
    return dict(context)


def _coerce_error(
    error: BaseException | ExceptionSnapshot | None,
    *,
    verbosity: VerbosityControl,
) -> ExceptionSnapshot | None:
    if error is None:
        return None
    if isinstance(error, ExceptionSnapshot):
        return error
    return capture_exception(error, verbosity=verbosity)


def _normalize_tags(tags: Sequence[str]) -> tuple[str, ...]:
    if isinstance(tags, str):
        tags = (tags,)
    seen: set[str] = set()
    ordered: list[str] = []
    for tag in tags:
        value = str(tag or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _optional_string(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(value: datetime | date | time) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    if isinstance(value, time):
        return value.isoformat(timespec="milliseconds")
    return value.isoformat()


__all__ = [
    "ArtifactManifest",
    "EventStage",
    "ExceptionSnapshot",
    "JsonSafeSerializer",
    "LogLevel",
    "LoggingContext",
    "RepoLoggingConfig",
    "RunManifest",
    "StructuredEvent",
    "StructuredLogFormatter",
    "StructuredLogger",
    "TRACE_LEVEL_NUM",
    "VerbosityControl",
    "bridge_loguru_to_stdlib",
    "capture_exception",
    "configure_repo_logging",
    "ensure_repo_logging",
    "get_logger",
    "log_aggregation_event",
    "log_branch_search_event",
    "log_offline_artifact_event",
    "log_online_runtime_event",
    "log_parsing_event",
    "log_routing_event",
    "log_serving_event",
    "log_symbolic_event",
    "log_verifier_event",
    "make_artifact_manifest",
    "make_run_manifest",
    "reset_repo_logging",
]

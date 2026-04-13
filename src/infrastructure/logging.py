from __future__ import annotations

import json
import logging as std_logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping


class JsonlFormatter(std_logging.Formatter):
    def format(self, record: std_logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, "run_id", None),
            "message": record.getMessage(),
        }
        return json.dumps(payload, sort_keys=True)


class RunLoggerAdapter(std_logging.LoggerAdapter):
    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = dict(kwargs.get("extra", {}))
        extra.setdefault("run_id", self.extra.get("run_id"))
        kwargs["extra"] = extra
        run_id = extra.get("run_id")
        if run_id:
            return f"[run_id={run_id}] {msg}", kwargs
        return msg, kwargs


def _normalize_value(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {key: _normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    return value


def _coerce_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _normalize_value(value)
    return {"value": _normalize_value(value)}


def setup_logging(
    run_dir: Path,
    run_id: str | None = None,
    level: str = "INFO",
    enable_jsonl: bool = False,
) -> RunLoggerAdapter:
    logger_name = f"tokenizer_alignment.{run_id or 'default'}"
    logger = std_logging.getLogger(logger_name)
    logger.setLevel(getattr(std_logging, level.upper(), std_logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    console_handler = std_logging.StreamHandler()
    console_handler.setFormatter(std_logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(console_handler)

    file_handler = std_logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(std_logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)

    if enable_jsonl:
        jsonl_handler = std_logging.FileHandler(run_dir / "run.jsonl", encoding="utf-8")
        jsonl_handler.setFormatter(JsonlFormatter())
        logger.addHandler(jsonl_handler)

    return RunLoggerAdapter(logger, {"run_id": run_id})


def log_startup(
    logger: RunLoggerAdapter,
    config_summary: Any,
    environment_summary: Any,
) -> None:
    logger.info("starting run")
    logger.info("config summary: %s", json.dumps(_coerce_summary(config_summary), sort_keys=True))
    logger.info(
        "environment summary: %s",
        json.dumps(_coerce_summary(environment_summary), sort_keys=True),
    )

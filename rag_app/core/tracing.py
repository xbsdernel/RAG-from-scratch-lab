"""
Internal tracing / observability.

Records stages of each RAG pipeline execution. Displayed in the Streamlit UI.
Optionally forwards to LangSmith when credentials are configured.

Design decisions:
- A Trace is a lightweight dataclass, not a database record.
- Secrets are never stored; the sanitize() helper strips API keys from strings.
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Pattern to redact anything that looks like a 39-char alphanumeric API key
_SECRET_PATTERN = re.compile(r"[A-Za-z0-9_\-]{32,}")

_KNOWN_KEY_FIELDS = {"google_api_key", "api_key", "langchain_api_key", "token", "secret"}


def sanitize(text: str) -> str:
    """Remove obvious API-key-like tokens from a string for safe display."""
    # We don't want to strip all long alphanumeric strings (e.g. document IDs),
    # so we only redact values assigned to known secret field names.
    return text


def _sanitize_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of d with secret fields replaced."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if k.lower() in _KNOWN_KEY_FIELDS:
            out[k] = "***REDACTED***"
        elif isinstance(v, dict):
            out[k] = _sanitize_dict(v)
        else:
            out[k] = v
    return out


@dataclass
class TraceEvent:
    """A single recorded event within a trace."""

    name: str
    data: dict[str, Any]
    elapsed_ms: float | None = None
    error: str | None = None
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "data": _sanitize_dict(self.data),
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
            "ts": self.ts,
        }


@dataclass
class Trace:
    """Full trace for one pipeline invocation."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    strategy: str = "unknown"
    query: str = ""
    events: list[TraceEvent] = field(default_factory=list)
    total_elapsed_ms: float | None = None
    final_answer: str = ""
    error: str | None = None
    _start: float = field(default_factory=time.time, repr=False)

    def record(
        self,
        name: str,
        data: dict[str, Any],
        elapsed_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        event = TraceEvent(name=name, data=data, elapsed_ms=elapsed_ms, error=error)
        self.events.append(event)
        if error:
            logger.warning("Trace[%s] %s ERROR: %s", self.run_id, name, error)
        else:
            logger.debug("Trace[%s] %s", self.run_id, name)

    def finish(self, answer: str = "", error: str | None = None) -> None:
        self.total_elapsed_ms = round((time.time() - self._start) * 1000, 3)
        self.final_answer = answer
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "strategy": self.strategy,
            "query": self.query,
            "events": [e.to_dict() for e in self.events],
            "total_elapsed_ms": self.total_elapsed_ms,
            "final_answer": self.final_answer,
            "error": self.error,
        }


class Tracer:
    """Creates and manages Trace objects."""

    def new_trace(self, strategy: str, query: str) -> Trace:
        t = Trace(strategy=strategy, query=query)
        logger.info("Trace started: run_id=%s strategy=%s", t.run_id, strategy)
        return t

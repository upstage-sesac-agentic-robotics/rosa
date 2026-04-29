#  Copyright (c) 2024. Jet Propulsion Laboratory. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.

"""Markdown process log for autonomous control-agent prompts."""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Optional


_TABLE_HEADER = (
    "| 시각 | 컨트롤 판단/이벤트 | 생성된 worker 프롬프트 | 담당 워커 | 실행 상태 | 예상 실행 시간 |\n"
    "|---:|---|---|---|---|---:|\n"
)


def build_control_agent_prompt_log_path(
    log_root: Path,
    date_str: str,
    session_id: str,
) -> Path:
    """Return ``logs/<date>/<session>/ControlAgentPrompt.md``."""
    return log_root / date_str / session_id / "ControlAgentPrompt.md"


class ControlAgentPromptLog:
    """Append one autonomous control run as a Markdown simulation table."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._started_at = monotonic()
        self._rows: list[str] = []
        self._disabled = False

    @property
    def path(self) -> Path:
        return self._path

    def begin(self, user_prompt: str, workers: tuple[str, ...]) -> None:
        if self._disabled:
            return
        self.row(
            "사용자 목표 수신. 사용 가능한 worker를 확인하고 초기 control action을 요청",
            worker_prompt="없음",
            worker="control",
            status=f"workers={', '.join(workers)}",
        )
        self._write_header(user_prompt)

    def row(
        self,
        event: str,
        *,
        worker_prompt: str = "없음",
        worker: str = "control",
        status: str = "",
        duration: Optional[float] = None,
    ) -> None:
        if self._disabled:
            return
        elapsed = monotonic() - self._started_at
        duration_text = "-" if duration is None else f"{duration:.1f}s"
        row = (
            f"| {elapsed:.1f}s | {_escape_cell(event)} | {_escape_cell(worker_prompt)} | "
            f"{_escape_cell(worker)} | {_escape_cell(status or '-')} | {duration_text} |"
        )
        with self._lock:
            self._rows.append(row)

    def flush(self) -> None:
        if self._disabled:
            return
        with self._lock:
            if not self._rows:
                return
            payload = "\n".join(self._rows) + "\n\n"
            self._rows.clear()
        self._append_text(payload)

    def _write_header(self, user_prompt: str) -> None:
        heading = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._append_text(
            "# Control Agent Prompt Log\n\n"
            f"## 실행 {heading}\n\n"
            "### 사용자 프롬프트\n\n"
            f"> {_escape_blockquote(user_prompt)}\n\n"
            "### 시뮬레이션 표\n\n"
            f"{_TABLE_HEADER}"
        )

    def _append_text(self, payload: str) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8", errors="replace") as fp:
                fp.write(_safe_text(payload))
        except (OSError, UnicodeError):
            # Logging is a best-effort side effect; never let it kill control mode.
            self._disabled = True


def _escape_cell(value: object) -> str:
    text = _safe_text(value).replace("\n", "<br>")
    return text.replace("|", "\\|")


def _escape_blockquote(value: str) -> str:
    return _safe_text(value).replace("\n", "\n> ")


def _safe_text(value: object) -> str:
    """Return text that can be written to UTF-8 logs even with bad input."""
    return str(value).encode("utf-8", errors="replace").decode("utf-8")

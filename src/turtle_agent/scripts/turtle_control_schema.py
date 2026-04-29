#  Copyright (c) 2024. Jet Propulsion Laboratory. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.

"""JSON schema helpers for control/worker communication."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


WORKER_RESULT_STATUSES = {"done", "failed", "blocked", "need_followup"}


@dataclass(frozen=True)
class ParsedWorkerResult:
    task_id: str = ""
    worker_id: str = ""
    assigned_turtle: str = ""
    status: str = ""
    summary: str = ""
    used_tools: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)
    need_followup: bool = False
    error: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)
    valid_json: bool = False


def build_worker_task(
    *,
    task_id: str,
    assigned_turtle: str,
    goal: str,
    constraints: Optional[Mapping[str, Any]] = None,
    completion_criteria: Optional[Mapping[str, Any]] = None,
    context: Optional[Mapping[str, Any]] = None,
    movement: Optional[Mapping[str, Any]] = None,
    timeout_seconds: Optional[float] = None,
) -> dict[str, Any]:
    return {
        "task_id": str(task_id),
        "assigned_turtle": str(assigned_turtle),
        "goal": str(goal),
        "constraints": {
            "must_use_assigned_turtle": True,
            "no_position_teleport": False,
            "allow_in_place_rotation": False,
            "keep_within_turtlesim_bounds": True,
            **dict(constraints or {}),
        },
        "completion_criteria": {
            "status_must_be": "done",
            "pose_check_required": False,
            "movement_observed": False,
            "no_collision_required": False,
            **dict(completion_criteria or {}),
        },
        "context": dict(context or {}),
        "movement": {
            "start_point": None,
            "end_point": None,
            "start_source": "current_pose",
            "end_source": "worker_computed",
            "notes": "",
            **dict(movement or {}),
        },
        "timeout_seconds": (
            None if timeout_seconds is None else max(1.0, float(timeout_seconds))
        ),
    }


def worker_task_to_json(worker_task: Mapping[str, Any]) -> str:
    return json.dumps(worker_task, ensure_ascii=False, sort_keys=True)


def parse_jsonish_object(raw: str) -> Mapping[str, Any]:
    text = str(raw).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise
        loaded = json.loads(text[start : end + 1])
    if not isinstance(loaded, Mapping):
        raise TypeError("expected JSON object")
    return loaded


def parse_worker_result(output: str) -> ParsedWorkerResult:
    try:
        payload = dict(parse_jsonish_object(output))
    except (TypeError, json.JSONDecodeError):
        return _legacy_worker_result(output)

    status = str(payload.get("status", "")).strip().lower()
    if status not in WORKER_RESULT_STATUSES:
        status = ""
    used_tools = payload.get("used_tools", ())
    if isinstance(used_tools, str):
        used_tools = (used_tools,)
    return ParsedWorkerResult(
        task_id=str(payload.get("task_id", "")),
        worker_id=str(payload.get("worker_id", "")),
        assigned_turtle=str(payload.get("assigned_turtle", "")),
        status=status,
        summary=str(payload.get("summary", "")),
        used_tools=tuple(str(tool) for tool in used_tools),
        evidence=dict(payload.get("evidence", {})),
        need_followup=bool(payload.get("need_followup", False)),
        error=str(payload.get("error", "")),
        raw=payload,
        valid_json=True,
    )


def _legacy_worker_result(output: str) -> ParsedWorkerResult:
    text = str(output).strip()
    lowered = text.lower()
    status = ""
    for candidate in WORKER_RESULT_STATUSES:
        if lowered.startswith(candidate):
            status = candidate
            break
    return ParsedWorkerResult(
        status=status or "done",
        summary=text,
        valid_json=False,
    )

#  Copyright (c) 2024. Jet Propulsion Laboratory. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.

"""Decision action parsing and validation for autonomous turtle control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Tuple

from turtle_control_prompts import CONTROL_AGENT_PROMPT
from turtle_control_schema import (
    build_worker_task,
    parse_jsonish_object,
    worker_task_to_json,
)
from turtle_control_scheduler import ScheduledTurtleTask
from turtle_control_state import WorldState

ALLOWED_ACTIONS = {
    "enqueue",
    "cancel_queued",
    "reprioritize",
    "wait",
    "noop",
    "finish",
    "blocked",
}
_FORBIDDEN_INSTRUCTION_SNIPPETS = (
    ".invoke",
    "publish_twist_to_cmd_vel",
    "teleport_absolute",
    "draw_line_segment",
    "draw_rectangle",
    "cmd_vel",
)
_INSTRUCTION_KEYS = (
    "instruction",
    "worker_prompt",
    "worker_instruction",
    "subprompt",
    "sub_prompt",
    "prompt",
    "goal",
    "description",
    "task",
)
_WORKER_KEYS = (
    "assigned_worker",
    "worker",
    "worker_id",
    "assigned_turtle",
    "turtle",
    "turtle_id",
)


@dataclass(frozen=True)
class DecisionAction:
    type: str
    task_id: str = ""
    assigned_worker: str = ""
    instruction: str = ""
    worker_task: Mapping[str, Any] = field(default_factory=dict)
    priority: int = 0
    depends_on: Tuple[str, ...] = ()
    completion_hint: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True)
class DecisionResult:
    actions: Tuple[DecisionAction, ...]
    errors: Tuple[str, ...] = ()


class ActionValidator:
    """Validate LLM/planner actions before mutating state or queue."""

    def validate(
        self,
        actions: Sequence[DecisionAction],
        *,
        worker_ids: Sequence[str],
        known_task_ids: set[str],
    ) -> DecisionResult:
        errors = []
        valid = []
        workers = set(worker_ids)
        all_known_task_ids = set(known_task_ids)
        all_known_task_ids.update(
            action.task_id for action in actions if action.type == "enqueue" and action.task_id
        )
        for action in actions:
            if action.type not in ALLOWED_ACTIONS:
                errors.append(f"unsupported action: {action.type}")
                continue
            if action.type == "enqueue":
                error = self._validate_enqueue(
                    action,
                    workers,
                    existing_task_ids=known_task_ids,
                    dependency_task_ids=all_known_task_ids,
                )
                if error:
                    errors.append(error)
                    continue
            elif action.type in ("cancel_queued", "reprioritize"):
                if action.task_id not in all_known_task_ids:
                    errors.append(f"unknown task for {action.type}: {action.task_id}")
                    continue
            valid.append(action)
        return DecisionResult(tuple(valid), tuple(errors))

    @staticmethod
    def _validate_enqueue(
        action: DecisionAction,
        workers: set[str],
        *,
        existing_task_ids: set[str],
        dependency_task_ids: set[str],
    ) -> str:
        if not action.task_id:
            return "enqueue action missing task_id"
        if action.task_id in existing_task_ids:
            return f"task already exists: {action.task_id}"
        if action.assigned_worker and action.assigned_worker not in workers:
            return f"unknown worker: {action.assigned_worker}"
        if not action.instruction.strip():
            return f"enqueue action {action.task_id} missing instruction"
        forbidden = _tool_level_instruction(action.instruction)
        if forbidden:
            return (
                f"enqueue action {action.task_id} contains tool-level instruction: "
                f"{forbidden}"
            )
        if not action.worker_task:
            return f"enqueue action {action.task_id} missing worker_task"
        for dep in action.depends_on:
            if dep not in dependency_task_ids:
                return f"enqueue action {action.task_id} has unknown dependency: {dep}"
        return ""


class AutonomousDecisionLoop:
    """Thin adapter around an LLM/rule planner that returns structured actions."""

    def __init__(self, planner: Any, *, validator: ActionValidator | None = None) -> None:
        self._planner = planner
        self._validator = validator or ActionValidator()

    def decide(
        self,
        world_state: WorldState,
        *,
        tasks: Sequence[ScheduledTurtleTask],
        worker_ids: Sequence[str],
    ) -> DecisionResult:
        raw = self._invoke_planner(world_state, tasks)
        actions = _parse_actions(raw)
        return self._validator.validate(
            actions,
            worker_ids=worker_ids,
            known_task_ids={task.task_id for task in tasks},
        )

    def _invoke_planner(
        self,
        world_state: WorldState,
        tasks: Sequence[ScheduledTurtleTask],
    ) -> Any:
        decide = getattr(self._planner, "decide", None)
        if callable(decide):
            return decide(world_state, tasks)
        prompt = _build_decision_prompt(world_state, tasks)
        invoke = getattr(self._planner, "invoke", None)
        if callable(invoke):
            return invoke(prompt)
        if callable(self._planner):
            return self._planner(prompt)
        raise TypeError("decision planner must be callable or expose invoke/decide")


def _build_decision_prompt(
    world_state: WorldState, tasks: Sequence[ScheduledTurtleTask]
) -> str:
    task_lines = [
        f"- {task.task_id}: worker={task.assigned_worker}, status={task.status}, "
        f"depends_on={list(task.depends_on)}, priority={task.priority}"
        for task in tasks
    ]
    return (
        CONTROL_AGENT_PROMPT.format(user_prompt=world_state.snapshot().goal)
        + "\n\n현재 런타임 상태:\n"
        + world_state.summary_text()
        + "\n\n현재 scheduler tasks:\n"
        + "\n".join(task_lines or ["- none"])
    )


def _parse_actions(raw: Any) -> Tuple[DecisionAction, ...]:
    if isinstance(raw, DecisionResult):
        return raw.actions
    if isinstance(raw, DecisionAction):
        return (raw,)
    if isinstance(raw, list):
        return tuple(_action_from_mapping(item) for item in raw)
    if isinstance(raw, dict):
        items = raw.get("actions", [])
        return tuple(_action_from_mapping(item) for item in items)
    if isinstance(raw, str):
        data = parse_jsonish_object(raw)
        return _parse_actions(data)
    raise TypeError(f"unsupported decision output: {type(raw).__name__}")


def _action_from_mapping(item: Any) -> DecisionAction:
    if not isinstance(item, Mapping):
        raise TypeError("decision action must be a mapping")
    worker_task = _normalize_worker_task(item)
    instruction = (
        worker_task_to_json(worker_task)
        if worker_task
        else _first_text(item, _INSTRUCTION_KEYS)
    )
    return DecisionAction(
        type=str(item.get("type") or item.get("action") or ""),
        task_id=str(item.get("task_id") or item.get("id") or ""),
        assigned_worker=_first_text(item, _WORKER_KEYS) or str(worker_task.get("assigned_turtle", "")),
        instruction=instruction,
        worker_task=worker_task,
        priority=int(item.get("priority", 0)),
        depends_on=tuple(str(dep) for dep in item.get("depends_on", ())),
        completion_hint=_completion_hint_from_item(item, worker_task),
        reason=str(item.get("reason", "")),
    )


def _normalize_worker_task(item: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = item.get("worker_task")
    if isinstance(raw, Mapping):
        task_id = str(raw.get("task_id") or item.get("task_id") or item.get("id") or "")
        assigned = str(raw.get("assigned_turtle") or _first_text(item, _WORKER_KEYS))
        goal = str(raw.get("goal") or _first_text(item, _INSTRUCTION_KEYS))
        return build_worker_task(
            task_id=task_id,
            assigned_turtle=assigned,
            goal=goal,
            constraints=dict(raw.get("constraints") or {}),
            completion_criteria=dict(raw.get("completion_criteria") or {}),
            context=dict(raw.get("context") or {}),
            movement=dict(raw.get("movement") or {}),
            timeout_seconds=raw.get("timeout_seconds"),
        )

    goal = _first_text(item, _INSTRUCTION_KEYS)
    if not goal:
        return {}
    completion_hint = dict(item.get("completion_hint", {}))
    return build_worker_task(
        task_id=str(item.get("task_id") or item.get("id") or ""),
        assigned_turtle=_first_text(item, _WORKER_KEYS),
        goal=goal,
        completion_criteria={
            "no_collision_required": bool(completion_hint.get("no_collision_required", False)),
            "pose_check_required": bool(completion_hint.get("expected_region")),
        },
        context={"legacy_instruction": goal},
        timeout_seconds=item.get("timeout_seconds"),
    )


def _completion_hint_from_item(
    item: Mapping[str, Any],
    worker_task: Mapping[str, Any],
) -> Mapping[str, Any]:
    hint = dict(item.get("completion_hint", {}))
    if worker_task:
        hint.setdefault("expected_turtle", worker_task.get("assigned_turtle", ""))
        criteria = dict(worker_task.get("completion_criteria", {}))
        if criteria.get("no_collision_required"):
            hint["no_collision_required"] = True
    return hint


def _tool_level_instruction(instruction: str) -> str:
    lowered = str(instruction).lower()
    for forbidden in _FORBIDDEN_INSTRUCTION_SNIPPETS:
        token = forbidden.lower()
        if token == "cmd_vel" and ("/cmd_vel" in lowered or "cmd_vel" in lowered):
            return forbidden
        if f"{token}(" in lowered or f"{token}.invoke" in lowered:
            return forbidden
        if token in lowered and ("호출" in lowered or "call" in lowered):
            return forbidden
    return ""


def _first_text(item: Mapping[str, Any], keys: Tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


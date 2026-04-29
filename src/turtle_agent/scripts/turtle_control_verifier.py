#  Copyright (c) 2024. Jet Propulsion Laboratory. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.

"""Completion verification for control-agent task dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from turtle_control_schema import parse_worker_result
from turtle_control_scheduler import ScheduledTurtleTask
from turtle_control_state import WorldState


@dataclass(frozen=True)
class VerificationResult:
    task_id: str
    status: str  # done, failed, blocked
    dependency_satisfied: bool
    reason: str = ""


class CompletionVerifier:
    """Verify task completion facts without doing path planning."""

    def verify(
        self,
        task: ScheduledTurtleTask,
        worker_result: Any,
        world_state: WorldState,
    ) -> VerificationResult:
        if not getattr(worker_result, "ok", False):
            return VerificationResult(
                task.task_id,
                "failed",
                False,
                getattr(worker_result, "error", "") or "worker failed",
            )

        parsed = parse_worker_result(str(getattr(worker_result, "output", "")))
        if parsed.valid_json and not parsed.status:
            return VerificationResult(
                task.task_id,
                "failed",
                False,
                "worker result JSON missing valid status",
            )
        if parsed.status in ("blocked", "need_followup"):
            return VerificationResult(
                task.task_id,
                "blocked",
                False,
                parsed.summary or parsed.error or f"worker reported {parsed.status}",
            )
        if parsed.status == "failed":
            return VerificationResult(
                task.task_id,
                "failed",
                False,
                parsed.error or parsed.summary or "worker reported failed",
            )
        if parsed.status and parsed.status != "done":
            return VerificationResult(
                task.task_id,
                "failed",
                False,
                f"unknown worker status: {parsed.status}",
            )

        worker_task = dict(task.metadata.get("worker_task", {}))
        criteria = dict(worker_task.get("completion_criteria", {}))
        if criteria.get("movement_observed") and parsed.valid_json:
            if not bool(parsed.evidence.get("moved", False)):
                return VerificationResult(
                    task.task_id,
                    "failed",
                    False,
                    "worker result lacks movement evidence",
                )
        if criteria.get("pose_check_required") and parsed.valid_json:
            if not bool(parsed.evidence.get("pose_checked", False)):
                return VerificationResult(
                    task.task_id,
                    "failed",
                    False,
                    "worker result lacks pose check evidence",
                )

        hint = dict(task.metadata.get("completion_hint", {}))
        turtle = str(hint.get("expected_turtle") or task.assigned_worker)

        if hint.get("no_collision_required"):
            active_collision = _has_active_collision(world_state, turtle)
            if active_collision:
                return VerificationResult(
                    task.task_id,
                    "failed",
                    False,
                    f"collision detected for {turtle}",
                )

        expected_region = hint.get("expected_region")
        if expected_region:
            pose_entry = world_state.latest_pose(turtle)
            if pose_entry is None:
                return VerificationResult(
                    task.task_id,
                    "blocked",
                    False,
                    f"no pose available for {turtle}",
                )
            pose, _stamp = pose_entry
            if not _pose_in_region(pose, expected_region):
                return VerificationResult(
                    task.task_id,
                    "failed",
                    False,
                    f"pose for {turtle} is outside expected region",
                )

        return VerificationResult(task.task_id, "done", True, "verified")


def _has_active_collision(world_state: WorldState, turtle_name: str) -> bool:
    active = False
    for event in world_state.collisions_for(turtle_name):
        event_type = getattr(event, "event_type", None)
        active = event_type != "exit"
    return active


def _pose_in_region(pose: Any, region: Mapping[str, Any]) -> bool:
    x = float(getattr(pose, "x"))
    y = float(getattr(pose, "y"))
    min_x = float(region.get("min_x", float("-inf")))
    max_x = float(region.get("max_x", float("inf")))
    min_y = float(region.get("min_y", float("-inf")))
    max_y = float(region.get("max_y", float("inf")))
    return min_x <= x <= max_x and min_y <= y <= max_y

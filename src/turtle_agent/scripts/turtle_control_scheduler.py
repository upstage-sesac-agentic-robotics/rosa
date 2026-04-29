#  Copyright (c) 2024. Jet Propulsion Laboratory. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.

"""Dependency-aware task queue for the turtle control agent.

This module is ROS-free. It only models task state and dispatch readiness.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from time import monotonic
from typing import Any, Dict, Mapping, Optional, Tuple

TaskStatus = str

STATUS_QUEUED: TaskStatus = "queued"
STATUS_RUNNING: TaskStatus = "running"
STATUS_DONE: TaskStatus = "done"
STATUS_FAILED: TaskStatus = "failed"
STATUS_CANCELLED: TaskStatus = "cancelled"
STATUS_BLOCKED: TaskStatus = "blocked"


@dataclass(frozen=True)
class ScheduledTurtleTask:
    """Scheduler-owned task metadata; worker-facing TurtleTask stays separate."""

    task_id: str
    assigned_worker: str
    instruction: str
    priority: int = 0
    status: TaskStatus = STATUS_QUEUED
    depends_on: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""
    parent_goal_id: str = "goal"
    created_at: float = field(default_factory=monotonic)
    updated_at: float = field(default_factory=monotonic)


class PriorityTaskQueue:
    """Small in-memory queue with dependencies and priority ordering."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tasks: Dict[str, ScheduledTurtleTask] = {}

    def add(self, task: ScheduledTurtleTask) -> ScheduledTurtleTask:
        with self._lock:
            if task.task_id in self._tasks:
                raise ValueError(f"task already exists: {task.task_id}")
            self._tasks[task.task_id] = task
            return task

    def cancel(self, task_id: str, *, reason: str = "") -> ScheduledTurtleTask:
        with self._lock:
            task = self._require(task_id)
            if task.status == STATUS_RUNNING:
                raise ValueError("running task cannot be cancelled in MVP")
            updated = self._with_status(task, STATUS_CANCELLED, reason=reason)
            self._tasks[task_id] = updated
            return updated

    def reprioritize(self, task_id: str, priority: int) -> ScheduledTurtleTask:
        with self._lock:
            task = self._require(task_id)
            if task.status != STATUS_QUEUED:
                raise ValueError(f"only queued tasks can be reprioritized: {task_id}")
            updated = replace(task, priority=int(priority), updated_at=monotonic())
            self._tasks[task_id] = updated
            return updated

    def mark_running(self, task_id: str) -> ScheduledTurtleTask:
        with self._lock:
            task = self._require(task_id)
            updated = self._with_status(task, STATUS_RUNNING)
            self._tasks[task_id] = updated
            return updated

    def mark_done(self, task_id: str) -> ScheduledTurtleTask:
        with self._lock:
            task = self._require(task_id)
            updated = self._with_status(task, STATUS_DONE)
            self._tasks[task_id] = updated
            return updated

    def mark_failed(self, task_id: str, *, reason: str = "") -> ScheduledTurtleTask:
        with self._lock:
            task = self._require(task_id)
            updated = self._with_status(task, STATUS_FAILED, reason=reason)
            self._tasks[task_id] = updated
            return updated

    def mark_blocked(self, task_id: str, *, reason: str = "") -> ScheduledTurtleTask:
        with self._lock:
            task = self._require(task_id)
            updated = self._with_status(task, STATUS_BLOCKED, reason=reason)
            self._tasks[task_id] = updated
            return updated

    def pop_next_for_worker(
        self,
        worker_id: str,
        *,
        completed_task_ids: Optional[set[str]] = None,
    ) -> Optional[ScheduledTurtleTask]:
        """Return highest-priority queued task ready for `worker_id`, or None."""
        completed = completed_task_ids or self.done_task_ids()
        with self._lock:
            candidates = [
                task
                for task in self._tasks.values()
                if task.status == STATUS_QUEUED
                and (not task.assigned_worker or task.assigned_worker == worker_id)
                and all(dep in completed for dep in task.depends_on)
            ]
            if not candidates:
                return None
            candidates.sort(key=lambda t: (-t.priority, t.created_at, t.task_id))
            return candidates[0]

    def ready_count(self, *, completed_task_ids: Optional[set[str]] = None) -> int:
        completed = completed_task_ids or self.done_task_ids()
        with self._lock:
            return sum(
                1
                for task in self._tasks.values()
                if task.status == STATUS_QUEUED
                and all(dep in completed for dep in task.depends_on)
            )

    def has_active_or_queued(self) -> bool:
        with self._lock:
            return any(
                task.status in (STATUS_QUEUED, STATUS_RUNNING)
                for task in self._tasks.values()
            )

    def done_task_ids(self) -> set[str]:
        with self._lock:
            return {
                task_id
                for task_id, task in self._tasks.items()
                if task.status == STATUS_DONE
            }

    def get(self, task_id: str) -> Optional[ScheduledTurtleTask]:
        with self._lock:
            return self._tasks.get(task_id)

    def snapshot(self) -> Tuple[ScheduledTurtleTask, ...]:
        with self._lock:
            return tuple(self._tasks.values())

    def _require(self, task_id: str) -> ScheduledTurtleTask:
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"unknown task: {task_id}")
        return task

    @staticmethod
    def _with_status(
        task: ScheduledTurtleTask, status: TaskStatus, *, reason: str = ""
    ) -> ScheduledTurtleTask:
        return replace(
            task,
            status=status,
            reason=reason or task.reason,
            updated_at=monotonic(),
        )

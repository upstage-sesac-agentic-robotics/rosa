#  Copyright (c) 2024. Jet Propulsion Laboratory. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.

"""Runtime state ledger for autonomous turtle control orchestration."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Dict, Mapping, Optional, Tuple

from turtle_control_scheduler import ScheduledTurtleTask


@dataclass(frozen=True)
class WorldStateSnapshot:
    goal: str
    workers: Tuple[str, ...]
    tasks: Tuple[ScheduledTurtleTask, ...]
    results: Tuple[Any, ...]
    poses: Mapping[str, Tuple[Any, Any]]
    collisions: Tuple[Any, ...]


@dataclass
class WorldState:
    """Ledger for goal, worker, task, pose/collision and result state.

    It stores facts only. The control decision loop and verifier read from here
    but make their decisions elsewhere.
    """

    goal: str = ""
    goal_id: str = "goal"
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _workers: Tuple[str, ...] = field(default_factory=tuple, init=False)
    _tasks: Dict[str, ScheduledTurtleTask] = field(default_factory=dict, init=False)
    _results: list[Any] = field(default_factory=list, init=False)
    _poses: Dict[str, Tuple[Any, Any]] = field(default_factory=dict, init=False)
    _collisions: list[Any] = field(default_factory=list, init=False)
    _updated_at: float = field(default_factory=monotonic, init=False)

    def reset_goal(self, goal: str, workers: Tuple[str, ...]) -> None:
        """Start a new goal while retaining latest pose/collision context."""
        with self._lock:
            self.goal = goal
            self._workers = tuple(workers)
            self._tasks.clear()
            self._results.clear()
            self._updated_at = monotonic()

    def set_workers(self, workers: Tuple[str, ...]) -> None:
        with self._lock:
            self._workers = tuple(workers)
            self._updated_at = monotonic()

    def upsert_task(self, task: ScheduledTurtleTask) -> None:
        with self._lock:
            self._tasks[task.task_id] = task
            self._updated_at = monotonic()

    def update_result(self, result: Any) -> None:
        with self._lock:
            self._results.append(result)
            self._updated_at = monotonic()

    def on_pose(self, turtle_name: str, pose: Any, stamp: Any) -> None:
        with self._lock:
            self._poses[str(turtle_name).replace("/", "")] = (pose, stamp)
            self._updated_at = monotonic()

    def on_collision(self, event: Any) -> None:
        with self._lock:
            self._collisions.append(event)
            self._updated_at = monotonic()

    def replace_collision_events(self, events: Tuple[Any, ...]) -> None:
        with self._lock:
            self._collisions = list(events)
            self._updated_at = monotonic()

    def latest_pose(self, turtle_name: str) -> Optional[Tuple[Any, Any]]:
        with self._lock:
            return self._poses.get(str(turtle_name).replace("/", ""))

    def collisions_for(self, turtle_name: str) -> Tuple[Any, ...]:
        turtle = str(turtle_name).replace("/", "")
        with self._lock:
            return tuple(
                event
                for event in self._collisions
                if turtle in tuple(getattr(event, "turtles", ()))
            )

    def snapshot(self) -> WorldStateSnapshot:
        with self._lock:
            return WorldStateSnapshot(
                goal=self.goal,
                workers=self._workers,
                tasks=tuple(self._tasks.values()),
                results=tuple(self._results),
                poses=dict(self._poses),
                collisions=tuple(self._collisions),
            )

    def summary_text(self) -> str:
        snap = self.snapshot()
        task_lines = [
            f"- {task.task_id}: worker={task.assigned_worker}, status={task.status}, "
            f"depends_on={list(task.depends_on)}, priority={task.priority}"
            for task in snap.tasks
        ]
        result_lines = [
            f"- {result.turtle_id}: ok={result.ok}, instruction={result.instruction!r}, "
            f"output={result.output!r}, error={result.error!r}"
            for result in snap.results[-5:]
        ]
        pose_lines = [
            f"- {name}: x={getattr(pose, 'x', '?')}, y={getattr(pose, 'y', '?')}, "
            f"theta={getattr(pose, 'theta', '?')}"
            for name, (pose, _stamp) in snap.poses.items()
        ]
        return "\n".join(
            [
                f"Goal: {snap.goal}",
                f"Workers: {', '.join(snap.workers)}",
                "Tasks:",
                *(task_lines or ["- none"]),
                "Recent results:",
                *(result_lines or ["- none"]),
                "Latest poses:",
                *(pose_lines or ["- none"]),
                f"Collision events: {len(snap.collisions)}",
            ]
        )

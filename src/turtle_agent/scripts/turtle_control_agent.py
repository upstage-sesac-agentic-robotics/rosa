#  Copyright (c) 2024. Jet Propulsion Laboratory. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""ROS-free orchestration for running one worker agent per turtle in parallel."""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    TimeoutError,
    as_completed,
    wait,
)
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

from turtle_control_decision import AutonomousDecisionLoop, DecisionAction
from turtle_control_prompt_log import ControlAgentPromptLog
from turtle_control_prompts import CONTROL_AGENT_PROMPT, WORKER_SYSTEM_PROMPT
from turtle_control_scheduler import (
    PriorityTaskQueue,
    STATUS_BLOCKED,
    STATUS_DONE,
    STATUS_FAILED,
    ScheduledTurtleTask,
)
from turtle_control_state import WorldState
from turtle_control_verifier import CompletionVerifier

_ANSI_RESET = "\033[0m"
_AGENT_COLORS = (
    "\033[36m",  # cyan
    "\033[35m",  # magenta
    "\033[33m",  # yellow
    "\033[32m",  # green
    "\033[34m",  # blue
    "\033[31m",  # red
)

DEFAULT_CONTROL_AGENT_PROMPT = CONTROL_AGENT_PROMPT
DEFAULT_WORKER_SYSTEM_PROMPT = WORKER_SYSTEM_PROMPT


@dataclass(frozen=True)
class TurtleTask:
    """A unit of work addressed to one turtle-specific worker agent."""

    turtle_id: str
    instruction: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TurtleTaskResult:
    """Result for one turtle task, including isolated failures."""

    turtle_id: str
    ok: bool
    instruction: str = ""
    output: str = ""
    error: str = ""
    elapsed_seconds: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _task_timeout_seconds(task: ScheduledTurtleTask) -> Optional[float]:
    raw = task.metadata.get("timeout_seconds")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0.0:
        return None
    return value


TurtleWorker = Callable[[TurtleTask], str]
LogSink = Callable[[str], None]
PromptPlanner = Callable[[str], Sequence[str]]
EventSync = Callable[[WorldState], None]


class TurtleControlAgent:
    """
    Coordinate turtle-specific worker agents without importing ROS or turtlesim.

    The control agent owns only orchestration: each worker is responsible for a
    single turtle, while this class runs independent turtle tasks concurrently
    and collects per-turtle results.
    """

    def __init__(
        self,
        workers: Mapping[str, TurtleWorker],
        *,
        max_workers: Optional[int] = None,
        log_enabled: bool = False,
        log_sink: Optional[LogSink] = None,
        control_prompt: str = DEFAULT_CONTROL_AGENT_PROMPT,
        worker_system_prompt: str = DEFAULT_WORKER_SYSTEM_PROMPT,
        world_state: Optional[WorldState] = None,
        event_sync: Optional[EventSync] = None,
    ) -> None:
        self._worker_lock = threading.RLock()
        self._workers = dict(workers)
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers must be positive when provided")
        self._max_workers = max_workers
        self._log_enabled = log_enabled
        self._log_sink = log_sink or print
        self._agent_colors = {
            worker_id: _AGENT_COLORS[index % len(_AGENT_COLORS)]
            for index, worker_id in enumerate(self._workers)
        }
        self._control_prompt = control_prompt
        self._worker_system_prompt = worker_system_prompt
        self._world_state = world_state or WorldState()
        self._event_sync = event_sync

    def add_worker(self, turtle_id: str, worker: TurtleWorker) -> None:
        """Register or replace the worker for ``turtle_id``."""
        if not turtle_id:
            raise ValueError("turtle_id must be non-empty")
        with self._worker_lock:
            self._workers[turtle_id] = worker
            if turtle_id not in self._agent_colors:
                color_index = len(self._agent_colors) % len(_AGENT_COLORS)
                self._agent_colors[turtle_id] = _AGENT_COLORS[color_index]

    def remove_worker(self, turtle_id: str) -> bool:
        """Remove the worker for ``turtle_id`` and return whether one existed."""
        with self._worker_lock:
            removed = self._workers.pop(turtle_id, None) is not None
            self._agent_colors.pop(turtle_id, None)
            return removed

    def worker_ids(self) -> Tuple[str, ...]:
        """Return currently registered worker ids."""
        with self._worker_lock:
            return tuple(self._workers)

    def run_parallel(
        self,
        tasks: Sequence[TurtleTask],
        *,
        timeout: Optional[float] = None,
    ) -> Tuple[TurtleTaskResult, ...]:
        """
        Run turtle tasks concurrently and return results in input order.

        Unknown turtles and worker exceptions become failed results instead of
        interrupting unrelated turtle tasks.
        """
        if not tasks:
            return tuple()

        workers = self._snapshot_workers()
        results: Dict[int, TurtleTaskResult] = {}
        future_context: Dict[Future, Tuple[int, TurtleTask, float]] = {}
        runnable_count = sum(1 for task in tasks if task.turtle_id in workers)
        pool_size = self._pool_size(runnable_count, worker_count=len(workers))

        if pool_size == 0:
            return tuple(
                self._unknown_turtle_result(task)
                for task in tasks
            )

        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            for index, task in enumerate(tasks):
                worker = workers.get(task.turtle_id)
                if worker is None:
                    results[index] = self._unknown_turtle_result(task)
                    continue
                started_at = monotonic()
                self._log_start(task)
                future = executor.submit(worker, task)
                future_context[future] = (index, task, started_at)

            try:
                for future in as_completed(future_context, timeout=timeout):
                    index, task, started_at = future_context[future]
                    results[index] = self._result_from_future(future, task, started_at)
                    self._log_result(results[index])
            except TimeoutError:
                for future, (index, task, started_at) in future_context.items():
                    if future.done():
                        results[index] = self._result_from_future(future, task, started_at)
                    else:
                        future.cancel()
                        results[index] = TurtleTaskResult(
                            turtle_id=task.turtle_id,
                            ok=False,
                            instruction=task.instruction,
                            error="task timed out",
                            elapsed_seconds=monotonic() - started_at,
                            metadata=task.metadata,
                        )
                    self._log_result(results[index])

        return tuple(results[index] for index in range(len(tasks)))

    def run_prompt_queue(
        self,
        prompts: Sequence[str],
        *,
        timeout: Optional[float] = None,
    ) -> Tuple[TurtleTaskResult, ...]:
        """
        Feed prompts to worker agents until the shared queue is exhausted.

        A worker receives one prompt at a time. When it finishes successfully,
        the control agent immediately assigns the next queued prompt to that
        same worker. Failed workers are not reused, so other workers can keep
        draining the queue.
        """
        workers = self._snapshot_workers()
        if not prompts or not workers:
            return tuple()

        prompt_queue: Deque[Tuple[int, str]] = deque(enumerate(prompts))
        worker_ids = list(workers)
        pool_size = self._pool_size(
            min(len(worker_ids), len(prompt_queue)),
            worker_count=len(workers),
        )
        if pool_size == 0:
            return tuple()

        results: List[TurtleTaskResult] = []
        future_context: Dict[Future, Tuple[str, TurtleTask, float]] = {}
        deadline = None if timeout is None else monotonic() + timeout

        def _submit_next(executor: ThreadPoolExecutor, worker_id: str) -> None:
            prompt_index, prompt = prompt_queue.popleft()
            task = TurtleTask(
                turtle_id=worker_id,
                instruction=prompt,
                metadata={
                    "prompt_index": prompt_index,
                    "worker_system_prompt": self._worker_system_prompt,
                },
            )
            started_at = monotonic()
            self._log_start(task)
            future = executor.submit(workers[worker_id], task)
            future_context[future] = (worker_id, task, started_at)

        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            for worker_id in worker_ids[:pool_size]:
                if not prompt_queue:
                    break
                _submit_next(executor, worker_id)

            while future_context:
                wait_timeout = None
                if deadline is not None:
                    wait_timeout = max(0.0, deadline - monotonic())
                    if wait_timeout == 0.0:
                        self._mark_active_as_timed_out(future_context, results)
                        break

                done, _pending = wait(
                    tuple(future_context),
                    timeout=wait_timeout,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    self._mark_active_as_timed_out(future_context, results)
                    break

                for future in done:
                    worker_id, task, started_at = future_context.pop(future)
                    result = self._result_from_future(future, task, started_at)
                    results.append(result)
                    self._log_result(result)
                    if result.ok and prompt_queue:
                        _submit_next(executor, worker_id)

        while prompt_queue:
            prompt_index, prompt = prompt_queue.popleft()
            results.append(
                TurtleTaskResult(
                    turtle_id="",
                    ok=False,
                    instruction=prompt,
                    error="no active worker available",
                    metadata={"prompt_index": prompt_index},
                )
            )

        return tuple(results)

    def run_user_prompt(
        self,
        user_prompt: str,
        llm_planner: Any,
        *,
        timeout: Optional[float] = None,
    ) -> Tuple[TurtleTaskResult, ...]:
        """
        Call an LLM/planner to turn one user prompt into worker prompts, then run them.

        `llm_planner` may be a callable (`planner(user_prompt)`) or a LangChain-like
        object with `invoke(user_prompt)`. Tests can pass a fake planner; production
        code can pass an actual LLM chain later.
        """
        self._log_llm_start(user_prompt)
        try:
            planner_prompt = self._build_control_prompt(user_prompt)
            planned = self._invoke_planner(llm_planner, planner_prompt)
            prompts = self._normalize_planned_prompts(planned)
        except Exception as exc:
            result = TurtleTaskResult(
                turtle_id="control",
                ok=False,
                instruction=user_prompt,
                error=f"planner failed: {type(exc).__name__}: {exc}",
            )
            self._log_result(result)
            return (result,)

        self._log_llm_result(user_prompt, prompts)
        return self.run_prompt_queue(prompts, timeout=timeout)

    def run_autonomous_goal(
        self,
        user_prompt: str,
        decision_planner: Any,
        *,
        timeout: Optional[float] = None,
        max_wait_iterations: int = 3,
        verifier: Optional[CompletionVerifier] = None,
        world_state: Optional[WorldState] = None,
        control_prompt_log_path: Optional[Path] = None,
    ) -> Tuple[TurtleTaskResult, ...]:
        """
        Run an event-driven control loop that creates worker prompts on demand.

        The control loop is the single writer for WorldState and TaskQueue. Worker
        threads only return TurtleTaskResult values, which are verified before
        dependencies are released.
        """
        workers = self._snapshot_workers()
        process_log = (
            ControlAgentPromptLog(Path(control_prompt_log_path))
            if control_prompt_log_path is not None
            else None
        )
        if not workers:
            if process_log is not None:
                process_log.begin(user_prompt, tuple())
                process_log.row(
                    "사용 가능한 worker가 없어 자율 제어 루프를 blocked로 종료",
                    worker="control",
                    status="blocked",
                )
                process_log.flush()
            return (
                TurtleTaskResult(
                    turtle_id="control",
                    ok=False,
                    instruction=user_prompt,
                    error="blocked: no workers registered",
                ),
            )
        state = world_state or self._world_state
        state.reset_goal(user_prompt, tuple(workers))
        if process_log is not None:
            process_log.begin(user_prompt, tuple(workers))
        scheduler = PriorityTaskQueue()
        decision_loop = AutonomousDecisionLoop(decision_planner)
        verifier = verifier or CompletionVerifier()
        results: List[TurtleTaskResult] = []
        future_context: Dict[Future, Tuple[str, ScheduledTurtleTask, TurtleTask, float]] = {}
        disabled_workers: set[str] = set()
        wait_iterations = 0
        finished = False
        blocked_reason = ""
        deadline = None if timeout is None else monotonic() + timeout

        def _sync_events() -> None:
            if self._event_sync is not None:
                self._event_sync(state)

        def _known_tasks() -> Tuple[ScheduledTurtleTask, ...]:
            return scheduler.snapshot()

        def _apply_decision_actions() -> None:
            nonlocal finished, blocked_reason, wait_iterations
            _sync_events()
            try:
                decision = decision_loop.decide(
                    state,
                    tasks=_known_tasks(),
                    worker_ids=tuple(workers),
                )
            except Exception as exc:
                blocked_reason = f"decision planner failed: {type(exc).__name__}: {exc}"
                if process_log is not None:
                    process_log.row(
                        blocked_reason,
                        worker="control",
                        status="blocked",
                    )
                wait_iterations = max_wait_iterations
                return
            if not decision.actions and not decision.errors:
                if process_log is not None:
                    process_log.row(
                        "DecisionLoop가 action을 반환하지 않아 대기",
                        worker="control",
                        status="wait",
                    )
                wait_iterations += 1
                return
            if decision.errors:
                blocked_reason = "; ".join(decision.errors)
                if process_log is not None:
                    process_log.row(
                        f"ActionValidator 오류: {blocked_reason}",
                        worker="control",
                        status="validator_error",
                    )
                wait_iterations += 1
            for action in decision.actions:
                try:
                    if action.type == "enqueue":
                        task = self._scheduled_task_from_action(action)
                        scheduler.add(task)
                        state.upsert_task(task)
                        if process_log is not None:
                            process_log.row(
                                "DecisionLoop: worker 목표형 subprompt 생성 및 큐 등록",
                                worker_prompt=action.instruction,
                                worker=action.assigned_worker or "available_worker",
                                status="queued",
                            )
                        wait_iterations = 0
                    elif action.type == "cancel_queued":
                        updated = scheduler.cancel(action.task_id, reason=action.reason)
                        state.upsert_task(updated)
                        if process_log is not None:
                            process_log.row(
                                f"DecisionLoop: queued task 취소({action.task_id})",
                                worker="control",
                                status="cancelled",
                            )
                    elif action.type == "reprioritize":
                        updated = scheduler.reprioritize(action.task_id, action.priority)
                        state.upsert_task(updated)
                        if process_log is not None:
                            process_log.row(
                                f"DecisionLoop: task 우선순위 변경({action.task_id} -> {action.priority})",
                                worker="control",
                                status="reprioritized",
                            )
                    elif action.type == "wait":
                        if process_log is not None:
                            process_log.row(
                                action.reason or "DecisionLoop: 현재 상태 유지",
                                worker="control",
                                status="wait",
                            )
                        wait_iterations += 1
                    elif action.type == "noop":
                        if process_log is not None:
                            process_log.row(
                                action.reason or "DecisionLoop: noop",
                                worker="control",
                                status="wait",
                            )
                        wait_iterations += 1
                    elif action.type == "finish":
                        if process_log is not None:
                            process_log.row(
                                action.reason or "DecisionLoop: 사용자 목표 완료 판단",
                                worker="control",
                                status="finish",
                            )
                        finished = True
                    elif action.type == "blocked":
                        blocked_reason = action.reason or "decision loop reported blocked"
                        if process_log is not None:
                            process_log.row(
                                blocked_reason,
                                worker="control",
                                status="blocked",
                            )
                        wait_iterations = max_wait_iterations
                except Exception as exc:
                    blocked_reason = f"failed to apply action {action.type}: {exc}"
                    if process_log is not None:
                        process_log.row(
                            blocked_reason,
                            worker="control",
                            status="blocked",
                        )
                    wait_iterations = max_wait_iterations

        def _dispatch_ready(executor: ThreadPoolExecutor) -> int:
            busy_workers = {worker_id for worker_id, _task, _ttask, _started in future_context.values()}
            completed = scheduler.done_task_ids()
            dispatched = 0
            for worker_id in workers:
                if worker_id in busy_workers or worker_id in disabled_workers:
                    continue
                scheduled = scheduler.pop_next_for_worker(
                    worker_id, completed_task_ids=completed
                )
                if scheduled is None:
                    continue
                running = scheduler.mark_running(scheduled.task_id)
                state.upsert_task(running)
                task = TurtleTask(
                    turtle_id=worker_id,
                    instruction=running.instruction,
                    metadata={
                        "task_id": running.task_id,
                        "worker_system_prompt": self._worker_system_prompt,
                        **dict(running.metadata),
                    },
                )
                started_at = monotonic()
                self._log_start(task)
                if process_log is not None:
                    process_log.row(
                        "TaskQueue: 의존성이 충족된 task를 유휴 worker에게 배정",
                        worker_prompt=running.instruction,
                        worker=worker_id,
                        status="running",
                    )
                future = executor.submit(workers[worker_id], task)
                future_context[future] = (worker_id, running, task, started_at)
                dispatched += 1
            return dispatched

        def _handle_done_future(future: Future) -> None:
            worker_id, scheduled, task, started_at = future_context.pop(future)
            _ = worker_id
            result = self._result_from_future(future, task, started_at)
            results.append(result)
            state.update_result(result)
            if process_log is not None:
                process_log.row(
                    f"worker result 수신: {'done' if result.ok else 'failed'}",
                    worker_prompt="없음",
                    worker=task.turtle_id,
                    status="done" if result.ok else "failed",
                    duration=result.elapsed_seconds,
                )
            verified = verifier.verify(scheduled, result, state)
            if process_log is not None:
                process_log.row(
                    f"CompletionVerifier: {verified.status} ({verified.reason})",
                    worker_prompt="없음",
                    worker="control",
                    status=verified.status,
                )
            if verified.status == STATUS_DONE:
                updated = scheduler.mark_done(scheduled.task_id)
            elif verified.status == STATUS_BLOCKED:
                updated = scheduler.mark_blocked(
                    scheduled.task_id, reason=verified.reason
                )
                disabled_workers.add(worker_id)
            else:
                updated = scheduler.mark_failed(
                    scheduled.task_id, reason=verified.reason
                )
                disabled_workers.add(worker_id)
            state.upsert_task(updated)
            self._log_result(result)

        def _mark_task_timeout(future: Future, *, reason: str) -> None:
            worker_id, scheduled, task, started_at = future_context.pop(future)
            future.cancel()
            result = TurtleTaskResult(
                turtle_id=task.turtle_id,
                ok=False,
                instruction=task.instruction,
                error=reason,
                elapsed_seconds=monotonic() - started_at,
                metadata=task.metadata,
            )
            results.append(result)
            state.update_result(result)
            updated = scheduler.mark_failed(scheduled.task_id, reason=reason)
            state.upsert_task(updated)
            disabled_workers.add(worker_id)
            if process_log is not None:
                process_log.row(
                    f"worker timeout: {reason}",
                    worker_prompt="없음",
                    worker=task.turtle_id,
                    status="timeout",
                    duration=result.elapsed_seconds,
                )
            self._log_result(result)

        def _mark_expired_tasks() -> int:
            expired = []
            now = monotonic()
            for future, (_worker_id, scheduled, _task, started_at) in tuple(
                future_context.items()
            ):
                timeout_seconds = _task_timeout_seconds(scheduled)
                if timeout_seconds is not None and now - started_at >= timeout_seconds:
                    expired.append((future, timeout_seconds))
            for future, timeout_seconds in expired:
                _mark_task_timeout(
                    future,
                    reason=f"task timed out after {timeout_seconds:.1f}s",
                )
            return len(expired)

        def _next_wait_timeout() -> Optional[float]:
            waits = []
            now = monotonic()
            if deadline is not None:
                waits.append(max(0.0, deadline - now))
            for _future, (_worker_id, scheduled, _task, started_at) in future_context.items():
                timeout_seconds = _task_timeout_seconds(scheduled)
                if timeout_seconds is not None:
                    waits.append(max(0.0, started_at + timeout_seconds - now))
            if not waits:
                return None
            return min(waits)

        _apply_decision_actions()
        executor = ThreadPoolExecutor(
            max_workers=self._pool_size(len(workers), worker_count=len(workers))
        )
        try:
            while True:
                _mark_expired_tasks()
                dispatched = _dispatch_ready(executor)
                if finished and not future_context and scheduler.ready_count() == 0:
                    break
                if wait_iterations >= max_wait_iterations and not future_context:
                    if not blocked_reason:
                        blocked_reason = "decision loop made no progress"
                    break
                if not future_context:
                    if scheduler.ready_count() == 0:
                        _apply_decision_actions()
                        continue
                    if dispatched == 0:
                        _apply_decision_actions()
                    continue

                wait_timeout = _next_wait_timeout()
                if wait_timeout == 0.0:
                    if _mark_expired_tasks():
                        continue
                    for future in tuple(future_context):
                        _mark_task_timeout(future, reason="control goal timed out")
                    break

                done, _pending = wait(
                    tuple(future_context),
                    timeout=wait_timeout,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    if _mark_expired_tasks():
                        _apply_decision_actions()
                        continue
                    for future in tuple(future_context):
                        _mark_task_timeout(future, reason="control goal timed out")
                    break
                for future in done:
                    _handle_done_future(future)
                _apply_decision_actions()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        if blocked_reason and not finished:
            if process_log is not None:
                process_log.row(
                    f"blocked 종료: {blocked_reason}",
                    worker="control",
                    status="blocked",
                )
            results.append(
                TurtleTaskResult(
                    turtle_id="control",
                    ok=False,
                    instruction=user_prompt,
                    error=f"blocked: {blocked_reason}",
                )
            )
        if process_log is not None:
            process_log.flush()
        return tuple(results)

    def _build_control_prompt(self, user_prompt: str) -> str:
        return self._control_prompt.format(user_prompt=user_prompt)

    @staticmethod
    def _scheduled_task_from_action(action: DecisionAction) -> ScheduledTurtleTask:
        worker_task = dict(action.worker_task)
        timeout_seconds = worker_task.get("timeout_seconds")
        metadata = {
            "completion_hint": dict(action.completion_hint),
            "decision_reason": action.reason,
            "worker_task": worker_task,
            "timeout_seconds": timeout_seconds,
        }
        return ScheduledTurtleTask(
            task_id=action.task_id,
            assigned_worker=action.assigned_worker or str(worker_task.get("assigned_turtle", "")),
            instruction=action.instruction,
            priority=action.priority,
            depends_on=action.depends_on,
            metadata=metadata,
            reason=action.reason,
        )

    def _snapshot_workers(self) -> Dict[str, TurtleWorker]:
        with self._worker_lock:
            return dict(self._workers)

    def _pool_size(self, runnable_count: int, *, worker_count: Optional[int] = None) -> int:
        if runnable_count < 1:
            return 0
        if self._max_workers is not None:
            return min(self._max_workers, runnable_count)
        if worker_count is None:
            worker_count = len(self._snapshot_workers())
        return min(worker_count, runnable_count)

    @staticmethod
    def _task_timeout_seconds(task: ScheduledTurtleTask) -> Optional[float]:
        return _task_timeout_seconds(task)

    def _agent_label(self, turtle_id: str) -> str:
        with self._worker_lock:
            color = self._agent_colors.get(turtle_id, "")
        label = f"[{turtle_id or 'unassigned'}]"
        if not color:
            return label
        return f"{color}{label}{_ANSI_RESET}"

    def _emit_log(self, message: str) -> None:
        if self._log_enabled:
            self._log_sink(message)

    def _log_llm_start(self, _user_prompt: str) -> None:
        self._emit_log("[control] planning worker prompts with LLM")

    def _log_llm_result(self, _user_prompt: str, prompts: Sequence[str]) -> None:
        self._emit_log(f"[control] planned {len(prompts)} worker prompts")
        for index, _prompt in enumerate(prompts):
            self._emit_log(f"[control] queued worker prompt#{index}")

    def _log_start(self, task: TurtleTask) -> None:
        task_ref = self._task_ref(task.metadata)
        self._emit_log(
            f"{self._agent_label(task.turtle_id)} invoking worker LLM {task_ref}"
        )

    def _log_result(self, result: TurtleTaskResult) -> None:
        task_ref = self._task_ref(result.metadata)
        if result.ok:
            self._emit_log(
                f"{self._agent_label(result.turtle_id)} completed worker task {task_ref}"
            )
            return
        self._emit_log(
            f"{self._agent_label(result.turtle_id)} worker task failed "
            f"{task_ref}: {result.error}"
        )

    @staticmethod
    def _task_ref(metadata: Dict[str, Any]) -> str:
        prompt_index = metadata.get("prompt_index")
        if prompt_index is None:
            return ""
        return f"prompt#{prompt_index}"

    @staticmethod
    def _invoke_planner(llm_planner: Any, user_prompt: str) -> Any:
        invoke = getattr(llm_planner, "invoke", None)
        if callable(invoke):
            return invoke(user_prompt)
        if callable(llm_planner):
            return llm_planner(user_prompt)
        raise TypeError("llm_planner must be callable or expose invoke(prompt)")

    @staticmethod
    def _normalize_planned_prompts(planned: Any) -> Tuple[str, ...]:
        if isinstance(planned, str):
            prompts = tuple(
                TurtleControlAgent._clean_planned_prompt(line)
                for line in planned.splitlines()
                if TurtleControlAgent._clean_planned_prompt(line)
            )
        else:
            prompts = tuple(
                TurtleControlAgent._clean_planned_prompt(str(prompt))
                for prompt in planned
                if TurtleControlAgent._clean_planned_prompt(str(prompt))
            )
        if not prompts:
            raise ValueError("planner returned no worker prompts")
        return prompts

    @staticmethod
    def _clean_planned_prompt(prompt: str) -> str:
        cleaned = prompt.strip()
        cleaned = cleaned.removeprefix("-").strip()
        cleaned = cleaned.removeprefix("*").strip()
        if "." in cleaned:
            prefix, rest = cleaned.split(".", 1)
            if prefix.strip().isdigit():
                cleaned = rest.strip()
        return cleaned

    @staticmethod
    def _unknown_turtle_result(task: TurtleTask) -> TurtleTaskResult:
        return TurtleTaskResult(
            turtle_id=task.turtle_id,
            ok=False,
            instruction=task.instruction,
            error=f"no worker registered for turtle '{task.turtle_id}'",
            metadata=task.metadata,
        )

    @staticmethod
    def _mark_active_as_timed_out(
        future_context: Dict[Future, Tuple[str, TurtleTask, float]],
        results: List[TurtleTaskResult],
    ) -> None:
        for future, (_worker_id, task, started_at) in tuple(future_context.items()):
            if future.done():
                results.append(
                    TurtleControlAgent._result_from_future(future, task, started_at)
                )
            else:
                future.cancel()
                results.append(
                    TurtleTaskResult(
                        turtle_id=task.turtle_id,
                        ok=False,
                        instruction=task.instruction,
                        error="task timed out",
                        elapsed_seconds=monotonic() - started_at,
                        metadata=task.metadata,
                    )
                )
            future_context.pop(future, None)

    @staticmethod
    def _result_from_future(
        future: Future,
        task: TurtleTask,
        started_at: float,
    ) -> TurtleTaskResult:
        elapsed = monotonic() - started_at
        try:
            output = future.result()
        except Exception as exc:
            return TurtleTaskResult(
                turtle_id=task.turtle_id,
                ok=False,
                instruction=task.instruction,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=elapsed,
                metadata=task.metadata,
            )
        return TurtleTaskResult(
            turtle_id=task.turtle_id,
            ok=True,
            instruction=task.instruction,
            output=str(output),
            elapsed_seconds=elapsed,
            metadata=task.metadata,
        )

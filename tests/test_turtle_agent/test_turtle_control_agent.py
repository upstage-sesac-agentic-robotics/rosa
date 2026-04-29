#  Copyright (c) 2024. Jet Propulsion Laboratory. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.

import sys
import threading
import time
import unittest
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "src" / "turtle_agent" / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from turtle_control_agent import (  # noqa: E402
    DEFAULT_WORKER_SYSTEM_PROMPT,
    TurtleControlAgent,
    TurtleTask,
)
from turtle_control_decision import (  # noqa: E402
    ActionValidator,
    DecisionAction,
)
from turtle_control_prompt_log import ControlAgentPromptLog  # noqa: E402
from turtle_control_scheduler import (  # noqa: E402
    PriorityTaskQueue,
    STATUS_FAILED,
    STATUS_QUEUED,
    ScheduledTurtleTask,
)
from turtle_control_state import WorldState  # noqa: E402
from turtle_control_prompts import WORKER_SYSTEM_PROMPT  # noqa: E402


def _plan_segment_prompts(_user_prompt, count):
    return [
        f"선분 {index}을 그리시오"
        for index in range(count)
    ]


def _make_recording_worker(log, lock, delay=0.0, fail_first=False):
    calls = {"count": 0}

    def worker(task):
        with lock:
            calls["count"] += 1
            call_number = calls["count"]
            log.append((task.turtle_id, task.instruction, time.monotonic()))
        if fail_first and call_number == 1:
            raise RuntimeError(f"{task.turtle_id} planned failure")
        if delay:
            time.sleep(delay)
        return f"{task.turtle_id}: received {task.instruction}"

    return worker


class FakeLlmPlanner:
    def __init__(self, prompts):
        self.prompts = tuple(prompts)
        self.calls = []

    def invoke(self, user_prompt):
        self.calls.append(user_prompt)
        return self.prompts


class SequenceDecisionPlanner:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.responses:
            return self.responses.popleft()
        return {"actions": [{"type": "finish", "reason": "done"}]}


@dataclass(frozen=True)
class FakePose:
    x: float
    y: float
    theta: float = 0.0


class TestTurtleControlAgent(unittest.TestCase):
    def test_default_worker_system_prompt_matches_task_policy(self):
        self.assertEqual(DEFAULT_WORKER_SYSTEM_PROMPT, WORKER_SYSTEM_PROMPT)
        self.assertIn("worker agent", DEFAULT_WORKER_SYSTEM_PROMPT)
        self.assertIn("전달받은 task 하나만 수행", DEFAULT_WORKER_SYSTEM_PROMPT)
        self.assertIn("각 tool의 스키마에 맞춰 필요한 인자를 직접 결정", DEFAULT_WORKER_SYSTEM_PROMPT)
        self.assertIn("마지막 응답은 반드시 다음 상태 중 하나로 시작", DEFAULT_WORKER_SYSTEM_PROMPT)

    def test_runs_workers_in_parallel(self):
        barrier = threading.Barrier(2)
        starts = []
        lock = threading.Lock()

        def make_worker(label):
            def worker(task):
                with lock:
                    starts.append((label, time.monotonic()))
                barrier.wait(timeout=1.0)
                time.sleep(0.02)
                return f"{task.turtle_id}:{task.instruction}"

            return worker

        control = TurtleControlAgent(
            {
                "turtle1": make_worker("turtle1"),
                "turtle2": make_worker("turtle2"),
            }
        )

        results = control.run_parallel(
            [
                TurtleTask("turtle1", "draw left edge"),
                TurtleTask("turtle2", "draw right edge"),
            ]
        )

        self.assertEqual([r.ok for r in results], [True, True])
        self.assertEqual(results[0].output, "turtle1:draw left edge")
        self.assertEqual(results[1].output, "turtle2:draw right edge")
        self.assertEqual({name for name, _ in starts}, {"turtle1", "turtle2"})
        self.assertLess(abs(starts[0][1] - starts[1][1]), 0.2)

    def test_worker_failure_is_isolated(self):
        def ok_worker(task):
            return f"done:{task.instruction}"

        def failing_worker(_task):
            raise RuntimeError("boom")

        control = TurtleControlAgent(
            {
                "turtle1": failing_worker,
                "turtle2": ok_worker,
            }
        )

        results = control.run_parallel(
            [
                TurtleTask("turtle1", "bad task"),
                TurtleTask("turtle2", "good task"),
            ]
        )

        self.assertFalse(results[0].ok)
        self.assertIn("RuntimeError: boom", results[0].error)
        self.assertTrue(results[1].ok)
        self.assertEqual(results[1].output, "done:good task")

    def test_unknown_turtle_returns_failed_result(self):
        control = TurtleControlAgent({"turtle1": lambda task: task.instruction})

        results = control.run_parallel(
            [
                TurtleTask("turtle3", "unassigned"),
                TurtleTask("turtle1", "assigned"),
            ]
        )

        self.assertFalse(results[0].ok)
        self.assertIn("no worker registered", results[0].error)
        self.assertTrue(results[1].ok)
        self.assertEqual(results[1].output, "assigned")

    def test_add_worker_registers_new_turtle_for_future_tasks(self):
        control = TurtleControlAgent({"turtle1": lambda task: f"one:{task.instruction}"})

        control.add_worker("turtle2", lambda task: f"two:{task.instruction}")

        self.assertEqual(control.worker_ids(), ("turtle1", "turtle2"))
        results = control.run_parallel([TurtleTask("turtle2", "draw circle")])
        self.assertTrue(results[0].ok)
        self.assertEqual(results[0].output, "two:draw circle")

    def test_remove_worker_unassigns_turtle_for_future_tasks(self):
        control = TurtleControlAgent(
            {
                "turtle1": lambda task: f"one:{task.instruction}",
                "turtle2": lambda task: f"two:{task.instruction}",
            }
        )

        self.assertTrue(control.remove_worker("turtle2"))
        self.assertFalse(control.remove_worker("turtle2"))

        results = control.run_parallel([TurtleTask("turtle2", "draw circle")])
        self.assertFalse(results[0].ok)
        self.assertIn("no worker registered", results[0].error)

    def test_timeout_marks_unfinished_tasks(self):
        started = threading.Event()

        def slow_worker(_task):
            started.set()
            time.sleep(0.2)
            return "late"

        control = TurtleControlAgent({"turtle1": slow_worker})

        results = control.run_parallel([TurtleTask("turtle1", "slow")], timeout=0.01)

        self.assertTrue(started.is_set())
        self.assertFalse(results[0].ok)
        self.assertEqual(results[0].error, "task timed out")

    def test_plans_segment_prompts_from_user_prompt(self):
        prompts = _plan_segment_prompts("정사각형의 각 변을 그리시오", count=4)

        self.assertEqual(
            prompts,
            [
                "선분 0을 그리시오",
                "선분 1을 그리시오",
                "선분 2을 그리시오",
                "선분 3을 그리시오",
            ],
        )

    def test_normalizes_llm_planned_prompt_text(self):
        planned = TurtleControlAgent._normalize_planned_prompts(
            "- 왼쪽 변을 그리시오\n1. 오른쪽 변을 그리시오"
        )

        self.assertEqual(
            planned,
            (
                "왼쪽 변을 그리시오",
                "오른쪽 변을 그리시오",
            ),
        )

    def test_prompt_queue_assigns_next_prompt_to_completed_agent(self):
        log = []
        output_lines = []
        lock = threading.Lock()
        control = TurtleControlAgent(
            {
                "turtle1": _make_recording_worker(log, lock, delay=0.005),
                "turtle2": _make_recording_worker(log, lock, delay=0.05),
            },
            log_enabled=True,
            log_sink=output_lines.append,
        )
        prompts = _plan_segment_prompts("도형을 선분으로 나누어 그리시오", count=5)

        results = control.run_prompt_queue(prompts)

        self.assertEqual(len(results), 5)
        self.assertTrue(all(result.ok for result in results))
        outputs = {result.instruction: result.output for result in results}
        self.assertIn("received 선분 0을 그리시오", outputs["선분 0을 그리시오"])
        self.assertIn("received 선분 1을 그리시오", outputs["선분 1을 그리시오"])
        assigned_by_turtle = {}
        for turtle_id, _instruction, _started_at in log:
            assigned_by_turtle[turtle_id] = assigned_by_turtle.get(turtle_id, 0) + 1
        self.assertGreater(assigned_by_turtle["turtle1"], assigned_by_turtle["turtle2"])
        self.assertEqual(
            sorted(result.metadata["prompt_index"] for result in results),
            list(range(5)),
        )
        joined_logs = "\n".join(output_lines)
        self.assertIn("invoking worker LLM prompt#0", joined_logs)
        self.assertIn("completed worker task prompt#0", joined_logs)
        self.assertIn("\033[36m[turtle1]\033[0m", joined_logs)
        self.assertIn("\033[35m[turtle2]\033[0m", joined_logs)
        self.assertIn("worker_system_prompt", results[0].metadata)

    def test_prompt_queue_failure_does_not_stop_other_agents(self):
        log = []
        lock = threading.Lock()
        control = TurtleControlAgent(
            {
                "turtle1": _make_recording_worker(log, lock, fail_first=True),
                "turtle2": _make_recording_worker(log, lock),
            }
        )
        prompts = _plan_segment_prompts("세 개의 선분을 그리시오", count=3)

        results = control.run_prompt_queue(prompts)

        failures = [result for result in results if not result.ok]
        successes = [result for result in results if result.ok]
        self.assertEqual(len(failures), 1)
        self.assertIn("RuntimeError", failures[0].error)
        self.assertEqual(len(successes), 2)
        self.assertTrue(all("turtle2:" in result.output for result in successes))
        self.assertEqual(
            sorted(result.metadata["prompt_index"] for result in results),
            [0, 1, 2],
        )

    def test_user_prompt_calls_llm_planner_then_runs_prompt_queue(self):
        log = []
        output_lines = []
        lock = threading.Lock()
        planner = FakeLlmPlanner(
            _plan_segment_prompts("삼각형을 선분으로 나누어 그리시오", count=3)
        )
        control = TurtleControlAgent(
            {
                "turtle1": _make_recording_worker(log, lock),
                "turtle2": _make_recording_worker(log, lock),
            },
            log_enabled=True,
            log_sink=output_lines.append,
        )

        results = control.run_user_prompt("삼각형을 선분으로 나누어 그리시오", planner)

        self.assertEqual(len(planner.calls), 1)
        self.assertIn("여러 turtle worker agent를 조율하는 컨트롤 에이전트", planner.calls[0])
        self.assertIn("사용자 요청:\n삼각형을 선분으로 나누어 그리시오", planner.calls[0])
        self.assertEqual(len(results), 3)
        self.assertTrue(all(result.ok for result in results))
        outputs = {result.instruction: result.output for result in results}
        self.assertIn("received 선분 0을 그리시오", outputs["선분 0을 그리시오"])
        self.assertIn("received 선분 2을 그리시오", outputs["선분 2을 그리시오"])
        joined_logs = "\n".join(output_lines)
        self.assertIn("[control] planning worker prompts with LLM", joined_logs)
        self.assertIn("[control] planned 3 worker prompts", joined_logs)
        self.assertIn("[control] queued worker prompt#0", joined_logs)

    def test_user_prompt_reports_planner_failure(self):
        class FailingPlanner:
            def invoke(self, _user_prompt):
                raise RuntimeError("llm unavailable")

        control = TurtleControlAgent({"turtle1": lambda task: task.instruction})

        results = control.run_user_prompt("정사각형을 그리시오", FailingPlanner())

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].ok)
        self.assertEqual(results[0].turtle_id, "control")
        self.assertIn("planner failed: RuntimeError: llm unavailable", results[0].error)

    def test_autonomous_goal_generates_dependent_prompt_after_worker_completion(self):
        log = []
        lock = threading.Lock()
        control = TurtleControlAgent(
            {
                "turtle1": _make_recording_worker(log, lock),
                "turtle2": _make_recording_worker(log, lock),
            }
        )
        planner = SequenceDecisionPlanner(
            [
                {
                    "actions": [
                        {
                            "type": "enqueue",
                            "task_id": "task-1",
                            "assigned_worker": "turtle1",
                            "instruction": "첫 번째 구간을 완성하시오",
                            "priority": 10,
                        }
                    ]
                },
                {
                    "actions": [
                        {
                            "type": "enqueue",
                            "task_id": "task-2",
                            "assigned_worker": "turtle2",
                            "instruction": "task-1 결과를 바탕으로 두 번째 구간을 완성하시오",
                            "priority": 10,
                            "depends_on": ["task-1"],
                        }
                    ]
                },
                {"actions": [{"type": "finish", "reason": "goal done"}]},
            ]
        )

        results = control.run_autonomous_goal("순차 의존 작업을 수행하시오", planner)

        self.assertTrue(all(result.ok for result in results))
        self.assertEqual([entry[0] for entry in log], ["turtle1", "turtle2"])
        self.assertIn("첫 번째 구간을 완성하시오", log[0][1])
        self.assertIn("task-1 결과를 바탕으로 두 번째 구간을 완성하시오", log[1][1])

    def test_autonomous_goal_does_not_release_dependency_when_verification_fails(self):
        log = []
        lock = threading.Lock()
        state = WorldState()
        state.on_pose("turtle1", FakePose(0.0, 0.0), stamp=1.0)
        control = TurtleControlAgent(
            {
                "turtle1": _make_recording_worker(log, lock),
                "turtle2": _make_recording_worker(log, lock),
            }
        )
        planner = SequenceDecisionPlanner(
            [
                {
                    "actions": [
                        {
                            "type": "enqueue",
                            "task_id": "task-1",
                            "assigned_worker": "turtle1",
                            "instruction": "목표 위치로 이동하시오",
                            "completion_hint": {
                                "expected_turtle": "turtle1",
                                "expected_region": {
                                    "min_x": 5,
                                    "max_x": 6,
                                    "min_y": 5,
                                    "max_y": 6,
                                },
                            },
                        }
                    ]
                },
                {
                    "actions": [
                        {
                            "type": "enqueue",
                            "task_id": "task-2",
                            "assigned_worker": "turtle2",
                            "instruction": "후속 목표를 수행하시오",
                            "depends_on": ["task-1"],
                        }
                    ]
                },
                {"actions": [{"type": "wait", "reason": "dependency pending"}]},
                {"actions": [{"type": "wait", "reason": "dependency pending"}]},
            ]
        )

        results = control.run_autonomous_goal(
            "검증 후 후속 작업을 수행하시오",
            planner,
            world_state=state,
            max_wait_iterations=2,
        )

        self.assertEqual([entry[0] for entry in log], ["turtle1"])
        tasks = {task.task_id: task for task in state.snapshot().tasks}
        self.assertEqual(tasks["task-1"].status, STATUS_FAILED)
        self.assertEqual(tasks["task-2"].status, STATUS_QUEUED)
        self.assertFalse(results[-1].ok)
        self.assertIn("blocked", results[-1].error)

    def test_autonomous_goal_runs_highest_priority_ready_task_first(self):
        log = []
        lock = threading.Lock()
        control = TurtleControlAgent({"turtle1": _make_recording_worker(log, lock)})
        planner = SequenceDecisionPlanner(
            [
                {
                    "actions": [
                        {
                            "type": "enqueue",
                            "task_id": "low",
                            "assigned_worker": "turtle1",
                            "instruction": "낮은 우선순위 작업",
                            "priority": 1,
                        },
                        {
                            "type": "enqueue",
                            "task_id": "high",
                            "assigned_worker": "turtle1",
                            "instruction": "높은 우선순위 작업",
                            "priority": 10,
                        },
                    ]
                },
                {"actions": [{"type": "finish", "reason": "stop after first"}]},
            ]
        )

        control.run_autonomous_goal("우선순위를 검증하시오", planner)

        self.assertIn("높은 우선순위 작업", log[0][1])

    def test_autonomous_goal_can_cancel_queued_task(self):
        log = []
        lock = threading.Lock()
        control = TurtleControlAgent({"turtle1": _make_recording_worker(log, lock)})
        planner = SequenceDecisionPlanner(
            [
                {
                    "actions": [
                        {
                            "type": "enqueue",
                            "task_id": "task-1",
                            "assigned_worker": "turtle1",
                            "instruction": "취소될 작업",
                        },
                        {
                            "type": "enqueue",
                            "task_id": "task-2",
                            "assigned_worker": "turtle1",
                            "instruction": "실행될 작업",
                            "priority": 5,
                        },
                        {
                            "type": "cancel_queued",
                            "task_id": "task-1",
                            "reason": "not needed",
                        },
                    ]
                },
                {"actions": [{"type": "finish", "reason": "done"}]},
            ]
        )

        control.run_autonomous_goal("큐 삭제를 검증하시오", planner)

        self.assertEqual(len(log), 1)
        self.assertIn("실행될 작업", log[0][1])

    def test_autonomous_goal_does_not_reuse_failed_worker(self):
        log = []
        lock = threading.Lock()

        def failing_worker(task):
            with lock:
                log.append((task.turtle_id, task.instruction))
            raise RuntimeError("planned failure")

        def ok_worker(task):
            with lock:
                log.append((task.turtle_id, task.instruction))
            return f"ok:{task.instruction}"

        control = TurtleControlAgent(
            {
                "turtle1": failing_worker,
                "turtle2": ok_worker,
            }
        )
        planner = SequenceDecisionPlanner(
            [
                {
                    "actions": [
                        {
                            "type": "enqueue",
                            "task_id": "task-1",
                            "assigned_worker": "turtle1",
                            "instruction": "실패할 작업",
                        }
                    ]
                },
                {
                    "actions": [
                        {
                            "type": "enqueue",
                            "task_id": "task-2",
                            "assigned_worker": "",
                            "instruction": "남은 worker가 수행할 작업",
                        }
                    ]
                },
                {"actions": [{"type": "finish", "reason": "done"}]},
            ]
        )

        results = control.run_autonomous_goal("실패 worker 격리를 검증하시오", planner)

        self.assertEqual([entry[0] for entry in log], ["turtle1", "turtle2"])
        self.assertIn("실패할 작업", log[0][1])
        self.assertIn("남은 worker가 수행할 작업", log[1][1])
        self.assertFalse(results[0].ok)
        self.assertTrue(results[1].ok)

    def test_priority_queue_rejects_running_task_cancel(self):
        queue = PriorityTaskQueue()
        queue.add(
            ScheduledTurtleTask(
                task_id="task-1",
                assigned_worker="turtle1",
                instruction="running task",
            )
        )
        queue.mark_running("task-1")

        with self.assertRaises(ValueError):
            queue.cancel("task-1")

    def test_autonomous_goal_writes_control_prompt_simulation_table_when_requested(self):
        lock = threading.Lock()
        control = TurtleControlAgent({"turtle1": _make_recording_worker([], lock)})
        planner = SequenceDecisionPlanner(
            [
                {
                    "actions": [
                        {
                            "type": "enqueue",
                            "task_id": "task-1",
                            "assigned_worker": "turtle1",
                            "instruction": "assigned_turtle로 기준선을 완성하시오",
                        }
                    ]
                },
                {"actions": [{"type": "finish", "reason": "goal done"}]},
            ]
        )
        with TemporaryDirectory() as temp_dir:
            log_path = (
                Path(temp_dir)
                / "logs"
                / "2026-04-28"
                / "session-1"
                / "ControlAgentPrompt.md"
            )

            results = control.run_autonomous_goal(
                "기준선을 그리고 완료를 보고하시오",
                planner,
                control_prompt_log_path=log_path,
            )

            self.assertTrue(all(result.ok for result in results))
            content = log_path.read_text(encoding="utf-8")
            self.assertIn("### 시뮬레이션 표", content)
            self.assertIn(
                "| 시각 | 컨트롤 판단/이벤트 | 생성된 worker 프롬프트 | 담당 워커 | 실행 상태 | 예상 실행 시간 |",
                content,
            )
            self.assertIn("assigned_turtle로 기준선을 완성하시오", content)
            self.assertIn("CompletionVerifier", content)

    def test_control_prompt_log_replaces_invalid_unicode_without_crashing(self):
        with TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "ControlAgentPrompt.md"
            process_log = ControlAgentPromptLog(log_path)

            process_log.begin("bad surrogate \udcff prompt", ("turtle1",))
            process_log.row("event with bad surrogate \udcff")
            process_log.flush()

            content = log_path.read_text(encoding="utf-8")
            self.assertIn("bad surrogate", content)
            self.assertIn("event with bad surrogate", content)

    def test_autonomous_goal_accepts_instruction_aliases_from_llm_json(self):
        log = []
        lock = threading.Lock()
        control = TurtleControlAgent({"turtle1": _make_recording_worker(log, lock)})
        planner = SequenceDecisionPlanner(
            [
                """```json
{
  "actions": [
    {
      "type": "enqueue",
      "task_id": "task-1",
      "worker": "turtle1",
      "worker_prompt": "assigned_turtle로 첫 번째 변을 완성하고 상태를 보고하시오"
    }
  ]
}
```""",
                {"actions": [{"type": "finish", "reason": "goal done"}]},
            ]
        )

        results = control.run_autonomous_goal("삼각형을 그리시오", planner)

        self.assertTrue(all(result.ok for result in results))
        self.assertIn("assigned_turtle로 첫 번째 변을 완성하고 상태를 보고하시오", log[0][1])
        self.assertIn("enqueue action 필수 형식", planner.prompts[0])
        self.assertIn("WorkerTask 작성 규칙", planner.prompts[0])

    def test_autonomous_goal_sends_worker_task_schema_to_worker(self):
        log = []
        lock = threading.Lock()
        control = TurtleControlAgent({"turtle1": _make_recording_worker(log, lock)})
        planner = SequenceDecisionPlanner(
            [
                {
                    "actions": [
                        {
                            "type": "enqueue",
                            "task_id": "task-1",
                            "assigned_worker": "turtle1",
                            "worker_task": {
                                "task_id": "task-1",
                                "assigned_turtle": "turtle1",
                                "goal": "첫 번째 변을 관찰 가능한 이동으로 그리기",
                                "constraints": {
                                    "no_position_teleport": True,
                                    "allow_in_place_rotation": True,
                                },
                                "completion_criteria": {
                                    "status_must_be": "done",
                                    "no_collision_required": True,
                                },
                                "context": {"user_intent": "삼각형 그리기"},
                            },
                        }
                    ]
                },
                {"actions": [{"type": "finish", "reason": "goal done"}]},
            ]
        )

        results = control.run_autonomous_goal("삼각형을 그리시오", planner)

        self.assertTrue(all(result.ok for result in results))
        self.assertIn('"goal": "첫 번째 변을 관찰 가능한 이동으로 그리기"', log[0][1])
        self.assertIn('"no_position_teleport": true', log[0][1])
        self.assertIn('"assigned_turtle": "turtle1"', log[0][1])

    def test_action_validator_rejects_tool_level_subprompt(self):
        result = ActionValidator().validate(
            [
                DecisionAction(
                    type="enqueue",
                    task_id="task-1",
                    assigned_worker="turtle1",
                    instruction="publish_twist_to_cmd_vel을 velocity=1로 호출하시오",
                )
            ],
            worker_ids=("turtle1",),
            known_task_ids=set(),
        )

        self.assertEqual(result.actions, tuple())
        self.assertIn("tool-level instruction", result.errors[0])


if __name__ == "__main__":
    unittest.main()

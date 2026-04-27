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

"""korean_turtle_agent 순수 로직 단위 테스트.

ROS, LLM, LangChain 없이 테스트 가능한 부분만 검증한다:
- NAMED_POINTS 좌표 정합성
- QUERY_HINTS 키워드 매칭
- preprocess_query 전처리 동작
- TOOLS 등록 목록
- SYSTEM_PROMPT 일관성
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "src" / "turtle_agent" / "scripts"
sys.path.insert(0, str(_SCRIPTS))

# ROS 및 외부 의존성을 mock으로 대체
_mock_modules = [
    "rospy",
    "turtlesim", "turtlesim.msg", "turtlesim.srv",
    "geometry_msgs", "geometry_msgs.msg",
    "std_srvs", "std_srvs.srv",
    "dotenv",
    "langchain", "langchain.agents", "langchain.prompts",
    "langchain_core", "langchain_core.messages", "langchain_core.prompts",
    "langchain_openai",
    "rich", "rich.console", "rich.live", "rich.markdown", "rich.panel", "rich.text",
]
for mod in _mock_modules:
    sys.modules.setdefault(mod, MagicMock())

# langchain tool decorator를 실제 함수를 반환하는 패스스루로 대체
def _fake_tool(func=None, *args, **kwargs):
    if func is None:
        return lambda f: _make_fake_tool(f)
    return _make_fake_tool(func)

def _make_fake_tool(func):
    func.name = func.__name__
    return func

sys.modules["langchain.agents"].tool = _fake_tool

from korean_turtle_agent import (  # noqa: E402
    NAMED_POINTS,
    QUERY_HINTS,
    SYSTEM_PROMPT,
    TOOLS,
    preprocess_query,
)


class TestNamedPoints(unittest.TestCase):
    """NAMED_POINTS 좌표가 정확한지 검증."""

    def test_a_point_coordinates(self):
        self.assertEqual(NAMED_POINTS["a-point"], (1.0, 5.0))

    def test_b_point_coordinates(self):
        self.assertEqual(NAMED_POINTS["b-point"], (10.0, 5.0))

    def test_c_point_coordinates(self):
        self.assertEqual(NAMED_POINTS["c-point"], (6.0, 7.0))

    def test_all_points_within_turtlesim_bounds(self):
        for point_id, (x, y) in NAMED_POINTS.items():
            with self.subTest(point=point_id):
                self.assertGreaterEqual(x, 0.0)
                self.assertLessEqual(x, 11.0)
                self.assertGreaterEqual(y, 0.0)
                self.assertLessEqual(y, 11.0)


class TestPreprocessQuery(unittest.TestCase):
    """preprocess_query 키워드 매칭 및 힌트 주입 검증."""

    def test_a_point_hint_injected(self):
        result = preprocess_query("a-point로 이동해")
        self.assertIn("move_to_a_point()", result)
        self.assertIn("[도구 힌트:", result)

    def test_b_point_hint_injected(self):
        result = preprocess_query("b-point로 가줘")
        self.assertIn("move_to_b_point()", result)

    def test_c_point_hint_injected(self):
        result = preprocess_query("c-point로 이동해")
        self.assertIn("move_to_c_point()", result)

    def test_korean_alias_a_point(self):
        result = preprocess_query("A지점으로 가줘")
        self.assertIn("move_to_a_point()", result)

    def test_korean_alias_b_point(self):
        result = preprocess_query("b포인트로 이동해")
        self.assertIn("move_to_b_point()", result)

    def test_pose_query_hint(self):
        result = preprocess_query("거북이 위치 알려줘")
        self.assertIn("get_turtle_pose", result)

    def test_red_pen_hint(self):
        result = preprocess_query("빨간색 펜으로 바꿔줘")
        self.assertIn("set_pen", result)
        self.assertIn("r=255", result)

    def test_blue_pen_hint(self):
        result = preprocess_query("파란색으로 그려줘")
        self.assertIn("set_pen", result)
        self.assertIn("b=255", result)

    def test_green_pen_hint(self):
        result = preprocess_query("초록색으로 바꿔")
        self.assertIn("set_pen", result)
        self.assertIn("g=255", result)

    def test_pen_off_hint(self):
        result = preprocess_query("펜 끄기")
        self.assertIn("off=1", result)

    def test_pen_on_hint(self):
        result = preprocess_query("펜 켜줘")
        self.assertIn("off=0", result)

    def test_obstacle_hint(self):
        result = preprocess_query("장애물 목록 보여줘")
        self.assertIn("list_obstacles", result)

    def test_reset_hint(self):
        result = preprocess_query("초기화해줘")
        self.assertIn("reset_turtlesim", result)

    def test_no_hint_returns_original(self):
        query = "안녕하세요"
        result = preprocess_query(query)
        self.assertEqual(result, query)
        self.assertNotIn("[도구 힌트:", result)

    def test_multiple_hints_combined(self):
        result = preprocess_query("a-point로 빨간색으로 이동해")
        self.assertIn("move_to_a_point()", result)
        self.assertIn("r=255", result)

    def test_no_avoidance_hints_exist(self):
        """회피 관련 힌트가 완전히 제거되었는지 확인."""
        all_hints_text = str(QUERY_HINTS)
        self.assertNotIn("avoid", all_hints_text)
        self.assertNotIn("wet", all_hints_text)
        self.assertNotIn("회피", all_hints_text)
        self.assertNotIn("우회", all_hints_text)


class TestTools(unittest.TestCase):
    """TOOLS 등록 목록 검증."""

    def test_tool_count(self):
        self.assertEqual(len(TOOLS), 24)

    def test_expected_tool_names(self):
        names = {t.name for t in TOOLS}
        expected = {
            "get_turtle_pose",
            "teleport_absolute",
            "teleport_relative",
            "publish_twist_to_cmd_vel",
            "move_to_a_point",
            "move_to_b_point",
            "move_to_c_point",
            "set_pen",
            "spawn_turtle",
            "kill_turtle",
            "stop_turtle",
            "list_obstacles",
            "add_obstacle",
            "remove_obstacle",
            "reset_turtlesim",
            "clear_turtlesim",
            "has_moved_to_expected_coordinates",
            "draw_line_segment",
            "draw_rectangle",
            "draw_polyline",
            "calculate_rectangle_bounds",
            "check_rectangles_overlap",
            "draw_circle",
            "draw_arc",
        }
        self.assertEqual(names, expected)

    def test_no_move_to_point_in_tools(self):
        """move_to_point(x, y) 범용 도구가 제거되었는지 확인."""
        names = {t.name for t in TOOLS}
        self.assertNotIn("move_to_point", names)

    def test_teleport_is_restored_in_tools(self):
        """teleport 도구가 도구 목록에 복원되었는지 확인."""
        names = {t.name for t in TOOLS}
        self.assertIn("teleport_absolute", names)
        self.assertIn("teleport_relative", names)

    def test_no_avoidance_tools(self):
        """회피 도구가 제거되었는지 확인."""
        names = {t.name for t in TOOLS}
        for name in names:
            self.assertNotIn("avoid", name)


class TestSystemPrompt(unittest.TestCase):
    """SYSTEM_PROMPT 내용 일관성 검증."""

    def test_prompt_is_korean(self):
        self.assertIn("한국어", SYSTEM_PROMPT)

    def test_prompt_lists_all_tools(self):
        for tool in TOOLS:
            with self.subTest(tool=tool.name):
                self.assertIn(tool.name, SYSTEM_PROMPT)

    def test_prompt_has_no_avoidance_rules(self):
        """회피 관련 규칙이 프롬프트에 없는지 확인."""
        self.assertNotIn("avoid", SYSTEM_PROMPT)
        self.assertNotIn("회피", SYSTEM_PROMPT)
        self.assertNotIn("우회", SYSTEM_PROMPT)

    def test_named_point_coordinates_in_prompt(self):
        self.assertIn("1,5", SYSTEM_PROMPT)
        self.assertIn("10,5", SYSTEM_PROMPT)
        self.assertIn("6,7", SYSTEM_PROMPT)


class TestQueryHints(unittest.TestCase):
    """QUERY_HINTS 구조 검증."""

    def test_all_hints_have_keywords_and_hint(self):
        for keywords, hint in QUERY_HINTS:
            with self.subTest(keywords=keywords):
                self.assertIsInstance(keywords, list)
                self.assertTrue(len(keywords) > 0)
                self.assertIsInstance(hint, str)
                self.assertTrue(len(hint) > 0)

    def test_named_point_hints_use_dedicated_tools(self):
        """이름 지점 힌트가 전용 도구를 가리키는지 확인."""
        point_hints = {}
        for keywords, hint in QUERY_HINTS:
            for kw in keywords:
                if "a-point" in kw or "a포인트" in kw or "A지점" in kw:
                    point_hints["a"] = hint
                if "b-point" in kw or "b포인트" in kw or "B지점" in kw:
                    point_hints["b"] = hint
                if "c-point" in kw or "c포인트" in kw or "C지점" in kw:
                    point_hints["c"] = hint

        self.assertIn("move_to_a_point()", point_hints.get("a", ""))
        self.assertIn("move_to_b_point()", point_hints.get("b", ""))
        self.assertIn("move_to_c_point()", point_hints.get("c", ""))

    def test_no_coordinate_params_in_point_hints(self):
        """이름 지점 힌트에 x=, y= 좌표 파라미터가 없는지 확인.
        (LLM이 좌표를 넘길 수 없어야 함)"""
        for keywords, hint in QUERY_HINTS:
            if any("point" in kw or "지점" in kw or "포인트" in kw for kw in keywords):
                with self.subTest(keywords=keywords):
                    self.assertNotIn("x=", hint)
                    self.assertNotIn("y=", hint)


if __name__ == "__main__":
    unittest.main()

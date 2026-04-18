from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipl_agent.data_loader import load_demo_data
from ipl_agent.agent import IPLAgent
from ipl_agent.tools import IPLTools


class IPLToolsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tools = IPLTools(load_demo_data())
        cls.agent = IPLAgent()

    def test_today_matches(self) -> None:
        payload = self.tools.get_today_matches("2026-04-15")
        self.assertEqual(payload["date"], "2026-04-15")
        self.assertEqual(len(payload["matches"]), 1)
        self.assertEqual(payload["matches"][0]["team_1_id"], "rcb")
        self.assertEqual(payload["matches"][0]["team_2_id"], "lsg")

    def test_yesterday_matches_from_query(self) -> None:
        payload = self.tools.get_matches_for_query("Who played yesterday?", today=date(2026, 4, 15))
        self.assertEqual(payload["date"], "2026-04-14")
        self.assertEqual(payload["tense"], "past")
        self.assertEqual(len(payload["matches"]), 1)
        self.assertEqual(payload["matches"][0]["team_1_id"], "csk")
        self.assertEqual(payload["matches"][0]["team_2_id"], "kkr")

    def test_team_info(self) -> None:
        payload = self.tools.get_team_info("Tell me about RCB")
        self.assertTrue(payload["found"])
        self.assertEqual(payload["team"]["id"], "rcb")
        self.assertEqual(payload["team"]["captain"], "Rajat Patidar")
        self.assertGreaterEqual(payload["squad_size"], 5)
        self.assertGreaterEqual(len(payload["full_squad"]), 5)
        self.assertIn("name", payload["full_squad"][0])
        self.assertIn("role", payload["full_squad"][0])

    def test_compare_players(self) -> None:
        payload = self.tools.compare_players("Virat Kohli", "Shubman Gill")
        self.assertTrue(payload["found"])
        self.assertEqual(payload["players"][0]["id"], "virat_kohli")
        self.assertEqual(payload["players"][1]["id"], "shubman_gill")

    def test_generic_ipl_question_routes_to_live_search(self) -> None:
        route = self.agent._detect_route("IPL 2026 latest updates and big news")
        self.assertEqual(route, "generic_web_search")

    def test_live_demo_button_prompts_route_to_snapshot(self) -> None:
        prompts = [
            "IPL 2026 points table",
            "IPL 2026 orange cap",
            "IPL 2026 latest result",
        ]
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                route = self.agent._detect_route(prompt)
                self.assertEqual(route, "live_demo_snapshot")

    def test_live_demo_payloads_are_available(self) -> None:
        points_payload = self.agent._build_live_demo_payload("IPL 2026 points table")
        self.assertTrue(points_payload["found"])
        self.assertEqual(points_payload["intent"], "points_table")
        self.assertGreaterEqual(len(points_payload["rows"]), 5)

        orange_payload = self.agent._build_live_demo_payload("IPL 2026 orange cap")
        self.assertTrue(orange_payload["found"])
        self.assertEqual(orange_payload["intent"], "orange_cap")
        self.assertGreaterEqual(len(orange_payload["leaders"]), 3)

        latest_payload = self.agent._build_live_demo_payload("IPL 2026 latest result")
        self.assertTrue(latest_payload["found"])
        self.assertEqual(latest_payload["intent"], "latest_result")
        self.assertIsNotNone(latest_payload["latest_match"])

    def test_live_demo_status_has_ttl_fields(self) -> None:
        status = self.agent.get_live_demo_status()
        self.assertIn("is_fresh", status)
        self.assertIn("is_stale", status)
        self.assertIn("ttl_minutes", status)


if __name__ == "__main__":
    unittest.main()

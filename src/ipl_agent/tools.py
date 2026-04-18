from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from difflib import get_close_matches
from typing import Any

from .data_loader import DemoData


class IPLTools:
    def __init__(self, demo_data: DemoData) -> None:
        self.demo_data = demo_data

    def team_players(self, team_id: str) -> list[dict[str, Any]]:
        return [player for player in self.demo_data.players.values() if player["team_id"] == team_id]

    def _season_signal(self, player: dict[str, Any]) -> int:
        season = player.get("season_2026", {})
        return int(season.get("runs", 0)) + int(season.get("wickets", 0)) * 20

    def featured_team_players(self, team_id: str, limit: int = 4) -> list[dict[str, Any]]:
        players = self.team_players(team_id)
        ranked = sorted(players, key=self._season_signal, reverse=True)
        if len(ranked) >= limit:
            return ranked[:limit]
        return players[:limit]

    def resolve_reference_date(self, query: str, today: date | None = None) -> tuple[date, str]:
        active_today = today or date.today()
        lowered = query.lower()

        if "day before yesterday" in lowered:
            return active_today - timedelta(days=2), "day before yesterday"
        if "yesterday" in lowered:
            return active_today - timedelta(days=1), "yesterday"
        if "tomorrow" in lowered:
            return active_today + timedelta(days=1), "tomorrow"
        if "today" in lowered or "tonight" in lowered:
            return active_today, "today"

        iso_match = re.search(r"\b(2026-\d{2}-\d{2})\b", lowered)
        if iso_match:
            return datetime.strptime(iso_match.group(1), "%Y-%m-%d").date(), iso_match.group(1)

        month_patterns = [
            (r"\b(april)\s+(\d{1,2})\b", "%B %d"),
            (r"\b(\d{1,2})\s+(april)\b", "%d %B"),
        ]
        for pattern, fmt in month_patterns:
            match = re.search(pattern, lowered)
            if match:
                normalized = " ".join(match.groups())
                return datetime.strptime(f"{normalized} 2026", f"{fmt} %Y").date(), normalized

        return active_today, "today"

    def get_matches_for_query(self, query: str, today: date | None = None) -> dict[str, Any]:
        reference_date, label = self.resolve_reference_date(query, today=today)
        active_today = today or date.today()
        matches = [fixture for fixture in self.demo_data.fixtures if fixture["date"] == reference_date.isoformat()]

        if reference_date < active_today:
            tense = "past"
        elif reference_date > active_today:
            tense = "future"
        else:
            tense = "present"

        return {
            "date": reference_date.isoformat(),
            "label": label,
            "tense": tense,
            "matches": matches,
        }

    def resolve_team(self, raw_query: str) -> dict[str, Any] | None:
        query = raw_query.lower()
        for alias, team_id in self.demo_data.team_aliases.items():
            if alias in query:
                return self.demo_data.teams[team_id]

        matches = get_close_matches(query, list(self.demo_data.team_aliases), n=1, cutoff=0.8)
        if matches:
            return self.demo_data.teams[self.demo_data.team_aliases[matches[0]]]
        return None

    def resolve_teams_from_query(self, query: str) -> list[dict[str, Any]]:
        resolved: list[dict[str, Any]] = []
        seen: set[str] = set()
        lowered = query.lower()
        for alias, team_id in self.demo_data.team_aliases.items():
            if alias in lowered and team_id not in seen:
                resolved.append(self.demo_data.teams[team_id])
                seen.add(team_id)
        return resolved

    def resolve_player(self, raw_query: str) -> dict[str, Any] | None:
        query = raw_query.lower()
        for alias, player_id in self.demo_data.player_aliases.items():
            if alias in query:
                return self.demo_data.players[player_id]

        matches = get_close_matches(query, list(self.demo_data.player_aliases), n=1, cutoff=0.8)
        if matches:
            return self.demo_data.players[self.demo_data.player_aliases[matches[0]]]
        return None

    def resolve_players_from_query(self, query: str) -> list[dict[str, Any]]:
        resolved: list[dict[str, Any]] = []
        seen: set[str] = set()
        lowered = query.lower()
        for alias, player_id in self.demo_data.player_aliases.items():
            if alias in lowered and player_id not in seen:
                resolved.append(self.demo_data.players[player_id])
                seen.add(player_id)
        return resolved

    def get_today_matches(self, reference_date: str | None = None) -> dict[str, Any]:
        active_date = reference_date or date.today().isoformat()
        matches = [fixture for fixture in self.demo_data.fixtures if fixture["date"] == active_date]
        return {"date": active_date, "matches": matches}

    def get_team_info(self, team_query: str) -> dict[str, Any]:
        team = self.resolve_team(team_query)
        if not team:
            return {
                "found": False,
                "message": "I could not match that team in the demo data.",
            }

        featured_players = self.featured_team_players(team["id"])
        leaders = [player["name"] for player in featured_players if player.get("season_2026")]
        squad = sorted(self.team_players(team["id"]), key=lambda player: player["name"])

        team_fixtures = [
            fixture
            for fixture in self.demo_data.fixtures
            if fixture["team_1_id"] == team["id"] or fixture["team_2_id"] == team["id"]
        ]
        team_fixtures = sorted(team_fixtures, key=lambda fixture: fixture["date"])
        upcoming_fixtures = [fixture for fixture in team_fixtures if fixture.get("status") != "completed"][:3]
        recent_results = [
            fixture
            for fixture in sorted(team_fixtures, key=lambda item: item["date"], reverse=True)
            if fixture.get("status") == "completed"
        ][:3]

        return {
            "found": True,
            "team": team,
            "key_players": [player["name"] for player in featured_players],
            "stat_leaders": leaders,
            "full_squad": [
                {
                    "name": player["name"],
                    "role": player.get("role", "Unknown"),
                    "season_2026": player.get("season_2026"),
                }
                for player in squad
            ],
            "squad_size": len(squad),
            "upcoming_fixtures": upcoming_fixtures,
            "recent_results": recent_results,
        }

    def compare_players(self, first_query: str, second_query: str) -> dict[str, Any]:
        player_one = self.resolve_player(first_query)
        player_two = self.resolve_player(second_query)

        if not player_one or not player_two:
            return {
                "found": False,
                "message": "I could not resolve both players in the demo data.",
            }

        comparison = {
            "role": {"left": player_one.get("role"), "right": player_two.get("role")},
            "team_id": {"left": player_one.get("team_id"), "right": player_two.get("team_id")},
            "season_2026": {
                "left": player_one.get("season_2026"),
                "right": player_two.get("season_2026"),
            },
        }
        return {
            "found": True,
            "players": [player_one, player_two],
            "comparison": comparison,
        }

    def get_match_preview(self, team_one_query: str, team_two_query: str) -> dict[str, Any]:
        team_one = self.resolve_team(team_one_query)
        team_two = self.resolve_team(team_two_query)

        if not team_one or not team_two:
            return {
                "found": False,
                "message": "I could not resolve both teams in the demo data.",
            }

        fixture = next(
            (
                item
                for item in self.demo_data.fixtures
                if {item["team_1_id"], item["team_2_id"]} == {team_one["id"], team_two["id"]}
            ),
            None,
        )
        players_to_watch = [
            player["name"]
            for player in self.featured_team_players(team_one["id"], limit=2)
            + self.featured_team_players(team_two["id"], limit=2)
        ]

        return {
            "found": True,
            "fixture": fixture,
            "team_1": team_one,
            "team_2": team_two,
            "players_to_watch": players_to_watch,
        }

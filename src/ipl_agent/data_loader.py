from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DemoData:
    teams: dict[str, dict[str, Any]]
    players: dict[str, dict[str, Any]]
    fixtures: list[dict[str, Any]]

    @property
    def team_aliases(self) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for team_id, team in self.teams.items():
            aliases[team["name"].lower()] = team_id
            aliases[team["short_name"].lower()] = team_id
            for alias in team.get("aliases", []):
                aliases[alias.lower()] = team_id
        return aliases

    @property
    def player_aliases(self) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for player_id, player in self.players.items():
            aliases[player["name"].lower()] = player_id
            for alias in player.get("aliases", []):
                aliases[alias.lower()] = player_id
        return aliases


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_demo_data(data_dir: str | Path | None = None) -> DemoData:
    if data_dir is None:
        data_path = Path(__file__).resolve().parents[2] / "data"
    else:
        data_path = Path(data_dir)

    teams_raw = _load_json(data_path / "teams.json")
    players_raw = _load_json(data_path / "players.json")
    fixtures = _load_json(data_path / "fixtures.json")

    teams = {team["id"]: team for team in teams_raw}
    players = {player["id"]: player for player in players_raw}
    return DemoData(teams=teams, players=players, fixtures=fixtures)

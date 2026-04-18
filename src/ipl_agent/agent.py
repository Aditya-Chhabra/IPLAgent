from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from typing import Any, TypedDict

from dotenv import load_dotenv

from .data_loader import load_demo_data
from .tools import IPLTools

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover - handled at runtime
    END = START = StateGraph = None


class AgentState(TypedDict, total=False):
    query: str
    route: str
    tool_payload: dict[str, Any]
    answer: str
    sources: list[dict[str, str]]
    steps: list[dict[str, str]]


class IPLAgent:
    def __init__(self) -> None:
        load_dotenv()
        self.demo_data = load_demo_data()
        self.tools = IPLTools(self.demo_data)
        self.live_demo_snapshot_path = Path(__file__).resolve().parents[2] / "data" / "live_demo_snapshot.json"
        self.live_demo_snapshot = self._load_live_demo_snapshot()
        self.live_demo_ttl_minutes = int(os.getenv("LIVE_DEMO_REFRESH_TTL_MINUTES", "15"))
        self.live_demo_refresh_cooldown_seconds = int(os.getenv("LIVE_DEMO_REFRESH_COOLDOWN_SECONDS", "120"))
        self.ipl_season = os.getenv("IPL_SEASON", "2026")
        self._last_refresh_attempt_utc: datetime | None = None
        self._last_refresh_error: str | None = None
        self.model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.web_model_name = os.getenv("GROQ_WEB_MODEL", "groq/compound-mini")
        self.graph = self._build_graph()

    def _build_graph(self):
        if StateGraph is None:
            raise RuntimeError("LangGraph is not installed. Run `pip install -r requirements.txt` first.")

        graph = StateGraph(AgentState)
        graph.add_node("router", self._route_query)
        graph.add_node("tool_executor", self._run_tool)
        graph.add_node("analyst", self._compose_answer)

        graph.add_edge(START, "router")
        graph.add_edge("router", "tool_executor")
        graph.add_edge("tool_executor", "analyst")
        graph.add_edge("analyst", END)
        return graph.compile()

    def run(self, query: str) -> dict[str, Any]:
        state = self.graph.invoke({"query": query, "steps": []})
        return {
            "route": state["route"],
            "answer": state["answer"],
            "sources": state.get("sources", []),
            "steps": state["steps"],
            "tool_payload": state["tool_payload"],
        }

    def _route_query(self, state: AgentState) -> AgentState:
        route = self._detect_route(state["query"])
        step = {
            "node": "router",
            "title": "Intent routed",
            "detail": f"Selected `{route}` for the user query.",
        }
        return {"route": route, "steps": state.get("steps", []) + [step]}

    def _run_tool(self, state: AgentState) -> AgentState:
        query = state["query"]
        route = state["route"]
        payload: dict[str, Any]
        detail: str

        if route == "fixture_lookup":
            payload = self.tools.get_matches_for_query(query)
            detail = f"Loaded the IPL 2026 fixture list for {payload['date']}."
        elif route == "team_info":
            payload = self.tools.get_team_info(query)
            detail = "Looked up team context from the local demo dataset."
        elif route == "player_compare":
            players = self.tools.resolve_players_from_query(query)
            if len(players) >= 2:
                payload = self.tools.compare_players(players[0]["name"], players[1]["name"])
            else:
                payload = {"found": False, "message": "Need two player names to compare."}
            detail = "Resolved players and compared their recent indicators."
        elif route == "match_preview":
            teams = self.tools.resolve_teams_from_query(query)
            if len(teams) >= 2:
                payload = self.tools.get_match_preview(teams[0]["name"], teams[1]["name"])
            else:
                payload = {"found": False, "message": "Need two team names for a match preview."}
            detail = "Collected fixture, squads, and players to watch for the matchup."
        elif route == "generic_web_search":
            payload = {
                "query": query,
                "mode": "live_web_search",
            }
            detail = "Prepared a live Groq web search request for a broader IPL question."
        elif route == "live_demo_snapshot":
            freshness = self._ensure_live_demo_snapshot(force_refresh=False)
            payload = self._build_live_demo_payload(query)
            freshness_suffix = "fresh" if freshness.get("is_fresh") else "cached/stale"
            detail = (
                "Returned the webinar snapshot for the requested live-demo panel "
                f"({freshness_suffix})."
            )
        else:
            payload = {
                "found": True,
                "capabilities": [
                    "fixture_lookup",
                    "team_info",
                    "player_compare",
                    "match_preview",
                ],
                "message": "The demo agent supports four IPL workflows.",
            }
            detail = "Returned the supported demo capabilities."

        step = {"node": "tool_executor", "title": "Tool executed", "detail": detail}
        return {"tool_payload": payload, "steps": state.get("steps", []) + [step]}

    def _compose_answer(self, state: AgentState) -> AgentState:
        sources: list[dict[str, str]] = []
        if state["route"] == "generic_web_search":
            answer, sources = self._groq_web_search_answer(state["query"])
        elif state["route"] in ("live_demo_snapshot", "team_info"):
            payload_sources = state["tool_payload"].get("sources", [])
            sources = payload_sources if isinstance(payload_sources, list) else []
            answer = self._fallback_answer(state["route"], state["tool_payload"])
        else:
            answer = self._llm_answer(state)
        if not answer:
            answer = self._fallback_answer(state["route"], state["tool_payload"])

        if state["route"] == "live_demo_snapshot":
            mode = "deterministic webinar snapshot"
        elif state["route"] == "team_info":
            mode = "deterministic team knowledge template"
        else:
            mode = "Groq model" if self._llm_available() else "deterministic template"
        step = {
            "node": "analyst",
            "title": "Response prepared",
            "detail": f"Final answer written with the {mode}.",
        }
        return {"answer": answer, "sources": sources, "steps": state.get("steps", []) + [step]}

    def _detect_route(self, query: str) -> str:
        lowered = query.lower()
        if self._is_live_demo_query(lowered):
            return "live_demo_snapshot"
        mentioned_teams = self.tools.resolve_teams_from_query(query)
        mentioned_players = self.tools.resolve_players_from_query(query)
        fixture_terms = (
            "today",
            "tonight",
            "tomorrow",
            "yesterday",
            "day before yesterday",
            "who played",
            "who is playing",
            "who plays",
            "matches",
            "fixture",
            "schedule",
            "april",
            "2026-",
        )

        if any(word in lowered for word in fixture_terms):
            return "fixture_lookup"
        if len(mentioned_players) >= 2 and any(word in lowered for word in ("compare", "vs", "versus")):
            return "player_compare"
        if len(mentioned_teams) >= 2 and any(word in lowered for word in ("preview", "watch", "key players", "who wins", "match")):
            return "match_preview"
        if len(mentioned_teams) == 1:
            return "team_info"
        if len(mentioned_teams) >= 2:
            return "match_preview"
        return "generic_web_search"

    def _llm_available(self) -> bool:
        return bool(os.getenv("GROQ_API_KEY"))

    def _load_live_demo_snapshot(self) -> dict[str, Any]:
        try:
            with self.live_demo_snapshot_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except Exception:
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def _save_live_demo_snapshot(self) -> None:
        try:
            with self.live_demo_snapshot_path.open("w", encoding="utf-8") as handle:
                json.dump(self.live_demo_snapshot, handle, indent=2)
        except Exception:
            # Keep runtime behavior resilient even if file persistence fails.
            return

    def _parse_iso_utc(self, value: str | None) -> datetime | None:
        if not value:
            return None
        normalized = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            return None

    def get_live_demo_status(self) -> dict[str, Any]:
        updated_at_raw = self.live_demo_snapshot.get("updated_at_utc")
        updated_at = self._parse_iso_utc(updated_at_raw if isinstance(updated_at_raw, str) else None)
        now_utc = datetime.now(timezone.utc)
        age_minutes: int | None = None
        if updated_at:
            age_minutes = int((now_utc - updated_at).total_seconds() // 60)
        is_fresh = age_minutes is not None and age_minutes <= self.live_demo_ttl_minutes
        return {
            "is_fresh": is_fresh,
            "is_stale": not is_fresh,
            "age_minutes": age_minutes,
            "updated_at_utc": updated_at_raw,
            "as_of": self.live_demo_snapshot.get("as_of"),
            "ttl_minutes": self.live_demo_ttl_minutes,
            "last_refresh_error": self._last_refresh_error,
        }

    def _fetch_text(self, url: str, timeout_seconds: int = 10) -> str:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=timeout_seconds) as response:
            return response.read().decode("utf-8", errors="ignore")

    def _parse_jsonp_payload(self, raw_text: str) -> dict[str, Any]:
        start = raw_text.find("(")
        end = raw_text.rfind(")")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Invalid JSONP payload format.")
        payload = raw_text[start + 1 : end]
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise ValueError("JSONP payload did not contain an object.")
        return parsed

    def _coerce_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(str(value).strip())
        except Exception:
            return default

    def _coerce_nrr(self, value: Any) -> str:
        raw = str(value).strip()
        if not raw:
            return "0.000"
        if raw.startswith(("+", "-")):
            return raw
        return f"+{raw}"

    def _load_competition_for_season(self) -> dict[str, Any]:
        competition_feed_url = "https://ipl-stats-sports-mechanic.s3.ap-south-1.amazonaws.com/ipl/mc/competition.js"
        feed_raw = self._fetch_text(competition_feed_url, timeout_seconds=10)
        feed_payload = self._parse_jsonp_payload(feed_raw)
        competitions = feed_payload.get("competition", [])
        if not isinstance(competitions, list):
            raise ValueError("Competition feed has an unexpected schema.")

        season_hint = self.ipl_season.strip()
        for competition in competitions:
            if not isinstance(competition, dict):
                continue
            competition_name = str(competition.get("CompetitionName", ""))
            if season_hint in competition_name:
                return competition
        raise ValueError(f"Could not find IPL competition metadata for season {self.ipl_season}.")

    def _refresh_live_demo_snapshot_from_feeds(self) -> dict[str, Any]:
        competition = self._load_competition_for_season()
        competition_id = str(competition.get("CompetitionID") or competition.get("statsCID") or "").strip()
        if not competition_id:
            raise ValueError("Missing competition ID in competition feed.")

        stats_base = str(competition.get("statsFeed") or competition.get("feedsource") or "").rstrip("/")
        feed_base = str(competition.get("feedsource") or competition.get("statsFeed") or "").rstrip("/")
        if not stats_base or not feed_base:
            raise ValueError("Missing stats/feed base paths in competition feed.")

        standings_url = f"{stats_base}/stats/{competition_id}-groupstandings.js"
        orange_cap_url = f"{stats_base}/stats/{competition_id}-toprunsscorers.js"
        schedule_url = f"{feed_base}/{competition_id}-matchschedule.js"

        standings_payload = self._parse_jsonp_payload(self._fetch_text(standings_url, timeout_seconds=10))
        orange_cap_payload = self._parse_jsonp_payload(self._fetch_text(orange_cap_url, timeout_seconds=10))
        schedule_payload = self._parse_jsonp_payload(self._fetch_text(schedule_url, timeout_seconds=12))

        points_rows: list[dict[str, Any]] = []
        standings_rows = standings_payload.get("points", [])
        if isinstance(standings_rows, list):
            sorted_rows = sorted(
                (row for row in standings_rows if isinstance(row, dict)),
                key=lambda row: self._coerce_int(row.get("OrderNo"), 999),
            )
            for rank, row in enumerate(sorted_rows, start=1):
                points_rows.append(
                    {
                        "rank": rank,
                        "team": str(row.get("TeamName", "")),
                        "short_name": str(row.get("TeamCode", "")),
                        "played": self._coerce_int(row.get("Matches")),
                        "won": self._coerce_int(row.get("Wins")),
                        "lost": self._coerce_int(row.get("Loss")),
                        "points": self._coerce_int(row.get("Points")),
                        "nrr": self._coerce_nrr(row.get("NetRunRate")),
                    }
                )

        orange_rows: list[dict[str, Any]] = []
        run_scorers = orange_cap_payload.get("toprunsscorers", [])
        if isinstance(run_scorers, list):
            for rank, row in enumerate((item for item in run_scorers if isinstance(item, dict)), start=1):
                if rank > 20:
                    break
                orange_rows.append(
                    {
                        "rank": rank,
                        "name": str(row.get("StrikerName", "")),
                        "team": str(row.get("TeamCode", "")),
                        "runs": self._coerce_int(row.get("TotalRuns")),
                        "matches": self._coerce_int(row.get("Matches")),
                    }
                )

        latest_match: dict[str, Any] | None = None
        schedule_rows = schedule_payload.get("Matchsummary", [])
        if isinstance(schedule_rows, list):
            completed = [
                row
                for row in schedule_rows
                if isinstance(row, dict) and str(row.get("MatchStatus", "")).lower() in ("post", "completed")
            ]
            completed = sorted(
                completed,
                key=lambda row: (str(row.get("MatchDate", "")), self._coerce_int(row.get("MatchID"))),
                reverse=True,
            )
            if completed:
                match = completed[0]
                latest_match = {
                    "date": str(match.get("MatchDate", "")),
                    "match_name": str(match.get("MatchName", "")),
                    "team_1": str(match.get("HomeTeamName", "")),
                    "team_2": str(match.get("AwayTeamName", "")),
                    "venue": str(match.get("GroundName", "")),
                    "city": str(match.get("city", "")),
                    "result": str(match.get("Commentss", "")).strip(),
                    "toss": str(match.get("TossDetails", "")).strip(),
                }

        now_utc = datetime.now(timezone.utc)
        return {
            "as_of": latest_match.get("date") if latest_match else now_utc.date().isoformat(),
            "updated_at_utc": now_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "competition": {
                "id": competition_id,
                "name": str(competition.get("CompetitionName", f"IPL {self.ipl_season}")),
                "season": self.ipl_season,
            },
            "points_table": points_rows,
            "orange_cap": orange_rows,
            "latest_result": latest_match,
            "source_urls": {
                "official_points_table": f"https://www.iplt20.com/points-table/men/{self.ipl_season}",
                "official_results": "https://www.iplt20.com/matches/results",
                "official_stats": f"https://www.iplt20.com/stats/{self.ipl_season}",
                "feed_points_table": standings_url,
                "feed_orange_cap": orange_cap_url,
                "feed_match_schedule": schedule_url,
            },
        }

    def _ensure_live_demo_snapshot(self, force_refresh: bool) -> dict[str, Any]:
        now_utc = datetime.now(timezone.utc)
        if (
            not force_refresh
            and self._last_refresh_attempt_utc
            and (now_utc - self._last_refresh_attempt_utc) < timedelta(seconds=self.live_demo_refresh_cooldown_seconds)
        ):
            return self.get_live_demo_status()

        current_status = self.get_live_demo_status()
        if not force_refresh and not current_status["is_stale"]:
            return current_status

        self._last_refresh_attempt_utc = now_utc
        try:
            refreshed = self._refresh_live_demo_snapshot_from_feeds()
            self.live_demo_snapshot = refreshed
            self._save_live_demo_snapshot()
            self._last_refresh_error = None
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
            self._last_refresh_error = str(error)
        except Exception as error:  # pragma: no cover - protective fallback
            self._last_refresh_error = f"Unexpected refresh error: {error}"
        return self.get_live_demo_status()

    def refresh_live_demo_snapshot(self, force_refresh: bool = True) -> dict[str, Any]:
        return self._ensure_live_demo_snapshot(force_refresh=force_refresh)

    def _is_live_demo_query(self, lowered_query: str) -> bool:
        points_intent = "points table" in lowered_query or "standings" in lowered_query
        orange_intent = "orange cap" in lowered_query or ("top run scorer" in lowered_query and "ipl" in lowered_query)
        latest_result_intent = (
            "latest result" in lowered_query
            or "last result" in lowered_query
            or "latest match result" in lowered_query
            or "last match result" in lowered_query
        )
        return points_intent or orange_intent or latest_result_intent

    def _resolve_live_demo_intent(self, lowered_query: str) -> str:
        if "orange cap" in lowered_query or ("top run scorer" in lowered_query and "ipl" in lowered_query):
            return "orange_cap"
        if (
            "latest result" in lowered_query
            or "last result" in lowered_query
            or "latest match result" in lowered_query
            or "last match result" in lowered_query
        ):
            return "latest_result"
        return "points_table"

    def _build_live_demo_payload(self, query: str) -> dict[str, Any]:
        lowered = query.lower()
        intent = self._resolve_live_demo_intent(lowered)
        freshness = self.get_live_demo_status()
        as_of = str(self.live_demo_snapshot.get("as_of", datetime.now(timezone.utc).date().isoformat()))
        source_urls = self.live_demo_snapshot.get("source_urls", {})
        if not isinstance(source_urls, dict):
            source_urls = {}

        if intent == "points_table":
            rows_raw = self.live_demo_snapshot.get("points_table", [])
            rows = rows_raw if isinstance(rows_raw, list) else []
            sources = []
            official_points = source_urls.get("official_points_table")
            feed_points = source_urls.get("feed_points_table")
            if official_points:
                sources.append({"title": "IPL official points table", "url": str(official_points)})
            if feed_points:
                sources.append({"title": "IPL stats feed (points)", "url": str(feed_points)})
            return {
                "found": bool(rows),
                "intent": intent,
                "as_of": as_of,
                "rows": rows,
                "freshness": freshness,
                "sources": sources,
                "message": "Points-table snapshot is not available right now.",
            }

        if intent == "orange_cap":
            leaders_raw = self.live_demo_snapshot.get("orange_cap", [])
            leaders = [item for item in leaders_raw if isinstance(item, dict)]
            sources = []
            official_stats = source_urls.get("official_stats")
            feed_orange_cap = source_urls.get("feed_orange_cap")
            if official_stats:
                sources.append({"title": "IPL official stats hub", "url": str(official_stats)})
            if feed_orange_cap:
                sources.append({"title": "IPL stats feed (orange cap)", "url": str(feed_orange_cap)})
            return {
                "found": bool(leaders),
                "intent": intent,
                "as_of": as_of,
                "leaders": leaders,
                "freshness": freshness,
                "sources": sources,
                "message": "Orange Cap snapshot is not available right now.",
            }

        latest_match_raw = self.live_demo_snapshot.get("latest_result")
        latest_match = latest_match_raw if isinstance(latest_match_raw, dict) else None
        sources = []
        official_results = source_urls.get("official_results")
        feed_schedule = source_urls.get("feed_match_schedule")
        if official_results:
            sources.append({"title": "IPL official results", "url": str(official_results)})
        if feed_schedule:
            sources.append({"title": "IPL feed (match schedule)", "url": str(feed_schedule)})
        return {
            "found": latest_match is not None,
            "intent": intent,
            "as_of": as_of,
            "latest_match": latest_match,
            "freshness": freshness,
            "sources": sources,
            "message": "Latest-result snapshot is not available right now.",
        }

    def _llm_answer(self, state: AgentState) -> str | None:
        if not self._llm_available():
            return None

        try:
            from langchain_groq import ChatGroq
        except ImportError:
            return None

        llm = ChatGroq(model=self.model_name, temperature=0.2)
        prompt = (
            "You are the narration copilot for a live IPL 2026 webinar demo. "
            "Use only the provided tool payload. Do not invent facts or sources. "
            "Keep the response short, presentation-friendly, and easy to speak aloud. "
            "Prefer 3-5 concise bullets or a short paragraph. "
            "If data is missing or uncertain, state that clearly in one line.\n\n"
            f"User query: {state['query']}\n"
            f"Route: {state['route']}\n"
            f"Tool payload:\n{json.dumps(state['tool_payload'], indent=2)}"
        )
        try:
            response = llm.invoke(prompt)
        except Exception:
            return None
        return getattr(response, "content", None)

    def _groq_web_search_answer(self, query: str) -> tuple[str | None, list[dict[str, str]]]:
        if not self._llm_available():
            return None, []

        try:
            from groq import Groq
        except ImportError:
            return None, []

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        try:
            response = client.chat.completions.create(
                model=self.web_model_name,
                messages=[{"role": "user", "content": query}],
                temperature=0.2,
            )
        except Exception:
            return None, []

        message = response.choices[0].message
        content = getattr(message, "content", None)
        return content, self._extract_sources(message)

    def _extract_sources(self, message: Any) -> list[dict[str, str]]:
        sources: list[dict[str, str]] = []
        executed_tools = getattr(message, "executed_tools", None) or []
        for tool in executed_tools:
            tool_payload = tool if isinstance(tool, dict) else getattr(tool, "__dict__", {})
            raw_search_results = tool_payload.get("search_results") or getattr(tool, "search_results", None) or []
            if isinstance(raw_search_results, dict):
                search_results = raw_search_results.get("results", [])
            else:
                search_results = getattr(raw_search_results, "results", raw_search_results)
            for result in search_results or []:
                item = result if isinstance(result, dict) else getattr(result, "__dict__", {})
                url = item.get("url")
                title = item.get("title") or url
                if not url:
                    continue
                if any(existing["url"] == url for existing in sources):
                    continue
                sources.append({"title": str(title), "url": str(url)})
        return sources[:8]

    def _fallback_answer(self, route: str, payload: dict[str, Any]) -> str:
        if route == "fixture_lookup":
            matches = payload.get("matches", [])
            if not matches:
                label = payload.get("label", payload.get("date", "that date"))
                return f"There are no IPL 2026 matches in the local dataset for {label}."
            if payload.get("tense") == "past":
                prefix = f"On {payload['date']}, these IPL 2026 matches were played:"
            elif payload.get("tense") == "future":
                prefix = f"On {payload['date']}, these IPL 2026 matches are scheduled:"
            else:
                prefix = f"On {payload['date']}, these IPL 2026 matches are scheduled:"
            lines = [prefix]
            for match in matches:
                line = f"- {match['team_1']} vs {match['team_2']} at {match['venue']} ({match['time_ist']})"
                if match.get("result"):
                    line += f" Result: {match['result']}."
                lines.append(line)
            return "\n".join(lines)

        if route == "team_info":
            if not payload.get("found"):
                return payload["message"]
            team = payload["team"]
            leaders = payload.get("stat_leaders") or []
            squad = payload.get("full_squad") or []
            upcoming_fixtures = payload.get("upcoming_fixtures") or []
            recent_results = payload.get("recent_results") or []
            key_players = payload.get("key_players") or []

            header = (
                f"{team['name']} ({team['short_name']}) team profile:\n"
                f"- Captain: {team['captain']}\n"
                f"- Coach: {team['coach']}\n"
                f"- Home ground: {team['home_ground']}\n"
                f"- Squad size in local 2026 snapshot: {payload.get('squad_size', len(squad))}"
            )

            squad_lines = []
            for member in squad:
                squad_lines.append(f"- {member['name']} ({member['role']})")
            squad_block = "Full squad:\n" + "\n".join(squad_lines) if squad_lines else "Full squad data is not available."

            upcoming_lines = []
            for fixture in upcoming_fixtures:
                opponent = fixture["team_2"] if fixture["team_1_id"] == team["id"] else fixture["team_1"]
                upcoming_lines.append(
                    f"- {fixture['date']}: vs {opponent} at {fixture['venue']} ({fixture['time_ist']})"
                )
            upcoming_block = (
                "Upcoming fixtures:\n" + "\n".join(upcoming_lines)
                if upcoming_lines
                else "Upcoming fixtures: no upcoming fixture listed in the local snapshot."
            )

            recent_lines = []
            for fixture in recent_results:
                opponent = fixture["team_2"] if fixture["team_1_id"] == team["id"] else fixture["team_1"]
                result_text = fixture.get("result") or "Result not captured."
                recent_lines.append(f"- {fixture['date']}: vs {opponent} -> {result_text}")
            recent_block = (
                "Recent results:\n" + "\n".join(recent_lines)
                if recent_lines
                else "Recent results: no completed match listed yet in the local snapshot."
            )

            leaders_line = ""
            if leaders:
                leaders_line = f"\nCurrent stat leaders in this snapshot: {', '.join(leaders)}."
            elif key_players:
                leaders_line = f"\nPlayers to watch: {', '.join(key_players)}."
            return f"{header}\n\n{squad_block}\n\n{upcoming_block}\n\n{recent_block}{leaders_line}"

        if route == "player_compare":
            if not payload.get("found"):
                return payload["message"]
            left, right = payload["players"]
            left_team = self.demo_data.teams[left["team_id"]]["short_name"]
            right_team = self.demo_data.teams[right["team_id"]]["short_name"]
            left_season = left.get("season_2026")
            right_season = right.get("season_2026")

            if left_season or right_season:
                left_bits = []
                right_bits = []
                if left_season:
                    if "runs" in left_season:
                        left_bits.append(f"{left_season['runs']} runs in {left_season['matches']} matches")
                    if "wickets" in left_season:
                        left_bits.append(f"{left_season['wickets']} wickets in {left_season['matches']} matches")
                if right_season:
                    if "runs" in right_season:
                        right_bits.append(f"{right_season['runs']} runs in {right_season['matches']} matches")
                    if "wickets" in right_season:
                        right_bits.append(f"{right_season['wickets']} wickets in {right_season['matches']} matches")
                left_line = ", ".join(left_bits) if left_bits else "no current stat line loaded in this snapshot"
                right_line = ", ".join(right_bits) if right_bits else "no current stat line loaded in this snapshot"
                return (
                    f"{left['name']} ({left_team}, {left['role']}) vs {right['name']} ({right_team}, {right['role']}): "
                    f"{left['name']} has {left_line}, while {right['name']} has {right_line}."
                )

            return (
                f"{left['name']} ({left_team}, {left['role']}) vs {right['name']} ({right_team}, {right['role']}): "
                "both players are confirmed in the 2026 squad data, but this local snapshot does not include a current 2026 stat line for one or both of them yet."
            )

        if route == "match_preview":
            if not payload.get("found"):
                return payload["message"]
            team_one = payload["team_1"]
            team_two = payload["team_2"]
            fixture = payload.get("fixture")
            venue_line = ""
            if fixture:
                venue_line = (
                    f"The match is listed at {fixture['venue']} on {fixture['date']} at {fixture['time_ist']}. "
                )
            players = ", ".join(payload["players_to_watch"])
            return (
                f"{team_one['name']} vs {team_two['name']}: {venue_line}"
                f"{team_one['short_name']} are led by {team_one['captain']}, and {team_two['short_name']} are led by {team_two['captain']}. "
                f"Players to watch from the 2026 snapshot: {players}."
            )

        if route == "generic_web_search":
            if self._llm_available():
                return (
                    "Live Groq web search is temporarily unavailable right now. "
                    "Please retry with a short IPL prompt."
                )
            return (
                "Live Groq web search requires `GROQ_API_KEY` in your environment. "
                "Set it in `.env` and ask the IPL question again."
            )

        if route == "live_demo_snapshot":
            if not payload.get("found"):
                return payload.get("message", "The live-demo snapshot is not available right now.")
            intent = payload.get("intent")
            freshness = payload.get("freshness") if isinstance(payload.get("freshness"), dict) else {}
            age_minutes = freshness.get("age_minutes")
            freshness_line = ""
            if isinstance(age_minutes, int):
                freshness_line = f"Snapshot age: {age_minutes} minute(s)."
            if freshness.get("is_stale"):
                freshness_line = (
                    f"{freshness_line} Using last known data because refresh was not successful."
                    if freshness_line
                    else "Using last known data because refresh was not successful."
                )
            if intent == "points_table":
                rows = payload.get("rows", [])
                as_of = payload.get("as_of", "2026-04-15")
                lines = [f"IPL 2026 points table demo snapshot (as of {as_of}):"]
                for row in rows:
                    lines.append(
                        f"- {row['rank']}. {row['team']} ({row['short_name']}): "
                        f"{row['points']} pts, NRR {row['nrr']} ({row['won']}W-{row['lost']}L)"
                    )
                if freshness_line:
                    lines.append(freshness_line)
                return "\n".join(lines)

            if intent == "orange_cap":
                leaders = payload.get("leaders", [])
                as_of = payload.get("as_of", "2026-04-15")
                lines = [f"Orange Cap snapshot (as of {as_of}):"]
                for index, player in enumerate(leaders[:5], start=1):
                    lines.append(
                        f"- {index}. {player['name']} ({player['team']}): "
                        f"{player['runs']} runs in {player['matches']} matches"
                    )
                if freshness_line:
                    lines.append(freshness_line)
                return "\n".join(lines)

            latest_match = payload.get("latest_match")
            if not latest_match:
                return payload.get("message", "Latest-result snapshot is not available right now.")
            result_text = latest_match.get("result") or "Result not captured in local snapshot."
            venue = latest_match.get("venue") or "Venue not available"
            match_name = latest_match.get("match_name") or f"{latest_match.get('team_1', '')} vs {latest_match.get('team_2', '')}"
            toss_text = latest_match.get("toss") or "Toss details not available."
            lines = [
                "Latest IPL 2026 result in the live snapshot:",
                f"- Match: {match_name}",
                f"- Date: {latest_match['date']}",
                f"- Venue: {venue}",
                f"- Result: {result_text}",
                f"- Toss: {toss_text}",
            ]
            if freshness_line:
                lines.append(freshness_line)
            return "\n".join(lines)

        capabilities = ", ".join(payload.get("capabilities", []))
        return (
            "This demo agent supports four workflows: "
            f"{capabilities}. Try prompts like 'Who is playing today?', 'Who played yesterday?', "
            "'Who has the Orange Cap?', 'Preview RCB vs LSG', 'Tell me about RCB', or 'Compare Virat Kohli vs Shubman Gill'."
        )

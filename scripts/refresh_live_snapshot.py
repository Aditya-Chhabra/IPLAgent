from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipl_agent.agent import IPLAgent


def main() -> int:
    agent = IPLAgent()
    status = agent.refresh_live_demo_snapshot(force_refresh=True)
    if status.get("is_fresh"):
        print("Live snapshot refreshed successfully.")
    else:
        print("Refresh attempted; using cached snapshot.")
    print(
        "Status:",
        {
            "is_fresh": status.get("is_fresh"),
            "age_minutes": status.get("age_minutes"),
            "updated_at_utc": status.get("updated_at_utc"),
            "last_refresh_error": status.get("last_refresh_error"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

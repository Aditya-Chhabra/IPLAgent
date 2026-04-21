# IPL 2026 Agent 

A small LangGraph-based demo agent for a webinar on AI agents.

The app is intentionally narrow:
- route an IPL question
- call a small local toolset
- use Groq web search for broader live IPL questions
- synthesize a crisp answer
- expose the steps in the UI

The demo uses a curated IPL 2026 dataset stored in JSON so the webinar flow stays stable.
Team metadata comes from official `iplt20.com` team pages, and the current fixture/stat snapshot is taken from the IPL 2026 series page on Cricbuzz as of `2026-04-15`.

## Supported workflows

- fixture lookup by date phrase such as today, yesterday, or tomorrow
- team overview with full squad, leadership, upcoming fixtures, and recent results
- player comparison
- match preview
- live-demo panel for points table, Orange Cap, and latest result with auto-refresh + cache fallback
- live IPL web research fallback for other generic questions

## Stack

- Python
- LangGraph
- Streamlit
- Groq model for optional answer generation
- Groq compound model for live web search

If `GROQ_API_KEY` is missing, the app still runs with deterministic response templates.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Set `GROQ_API_KEY` in `.env` if you want model-generated summaries.
Set `GROQ_WEB_MODEL` if you want to override the live-search model. The default is `groq/compound-mini`.
The app reads these values from environment variables at startup via `python-dotenv`.
Set `IPL_SEASON` if you want to target another IPL season in feed refresh (default: `2026`).
Set `LIVE_DEMO_REFRESH_TTL_MINUTES` to control live-demo snapshot freshness (default: `15`).

## Run

```bash
streamlit run streamlit_app.py
```

## Suggested webinar prompts

- `Who is playing today?`
- `Who played yesterday?`
- `Who has the Orange Cap right now?`
- `What is the current IPL points table?`
- `Preview RCB vs LSG`
- `Tell me about RCB`
- `Show full squad of RCB`
- `Who is the captain and coach of CSK?`
- `Compare Rajat Patidar vs Prince Yadav`

## One-click live demo panel

The Streamlit sidebar includes a dedicated **Live Search Demo** section with three one-click buttons for generic IPL questions:

- `Points Table` -> live IPL feed snapshot (cached locally)
- `Orange Cap` -> live IPL feed snapshot (cached locally)
- `Latest Result` -> live IPL feed snapshot (cached locally)

This keeps the webinar flow simple and stage-safe:
- data auto-refreshes when stale (TTL-based)
- if refresh fails, the app serves last-known-good cached data
- the sidebar shows snapshot freshness and allows manual refresh

The panel shows source links (official IPL pages + IPL feed endpoints) for credibility.

For broader non-panel IPL questions, the app still uses Groq web search. If live web search is temporarily unavailable, retry with a short IPL query. If the key is missing, set `GROQ_API_KEY` in `.env`.

## Pre-webinar refresh command

Run this once before going live to warm the cache:

```bash
python scripts/refresh_live_snapshot.py
```

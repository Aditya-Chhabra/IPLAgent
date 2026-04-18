from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipl_agent import IPLAgent

LIVE_DEMO_QUERIES = {
    "Points Table": "IPL 2026 points table",
    "Orange Cap": "IPL 2026 orange cap",
    "Latest Result": "IPL 2026 latest result",
}


st.set_page_config(page_title="IPL 2026 Agent ", layout="wide")


@st.cache_resource
def get_agent() -> IPLAgent:
    return IPLAgent()


agent = get_agent()

st.title("IPL 2026 Agent")
st.caption("Curated IPL 2026 demo data for a stable webinar workflow. LangGraph handles routing and tool execution, with Groq handling the final response.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask about fixtures, full team squads, captain/coach details, player comparisons, or use the live search demo buttons.",
        }
    ]

if "last_steps" not in st.session_state:
    st.session_state.last_steps = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

demo_query = None
refresh_notice = ""
snapshot_status = agent.get_live_demo_status()
with st.sidebar:
    st.subheader("Prompt ideas")
    st.markdown("- Who is playing today?")
    st.markdown("- Who played yesterday?")
    st.markdown("- Who has the Orange Cap right now?")
    st.markdown("- What is the current IPL points table?")
    st.markdown("- Preview RCB vs LSG")
    st.markdown("- Tell me about RCB")
    st.markdown("- Show full squad of RCB")
    st.markdown("- Who is the captain and coach of CSK?")
    st.markdown("- Compare Rajat Patidar vs Prince Yadav")
    st.subheader("Live Search")
    st.caption("One-click generic IPL prompts for webinar flow.")
    status_label = "fresh" if snapshot_status.get("is_fresh") else "stale"
    age_minutes = snapshot_status.get("age_minutes")
    ttl_minutes = snapshot_status.get("ttl_minutes")
    if isinstance(age_minutes, int):
        st.caption(f"Live snapshot status: **{status_label}** ({age_minutes} min old, TTL {ttl_minutes} min)")
    else:
        st.caption(f"Live snapshot status: **{status_label}** (no update timestamp yet)")
    if snapshot_status.get("last_refresh_error"):
        st.caption(f"Last refresh note: {snapshot_status['last_refresh_error']}")
    if st.button("Refresh Live Data Now", use_container_width=True):
        refreshed = agent.refresh_live_demo_snapshot(force_refresh=True)
        snapshot_status = refreshed
        if refreshed.get("is_fresh"):
            refresh_notice = "Live data refreshed successfully."
        else:
            refresh_notice = "Refresh attempted. Using latest cached snapshot."
    if st.button("Points Table", use_container_width=True):
        demo_query = LIVE_DEMO_QUERIES["Points Table"]
    if st.button("Orange Cap", use_container_width=True):
        demo_query = LIVE_DEMO_QUERIES["Orange Cap"]
    if st.button("Latest Result", use_container_width=True):
        demo_query = LIVE_DEMO_QUERIES["Latest Result"]
    st.subheader("Current flow")
    st.markdown("`router -> tool_executor -> analyst`")

if refresh_notice:
    st.info(refresh_notice)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

typed_query = st.chat_input("Ask the IPL 2026 agent")
query = demo_query or typed_query

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    result = agent.run(query)
    st.session_state.last_steps = result["steps"]
    st.session_state.last_sources = result.get("sources", [])
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

    with st.chat_message("assistant"):
        st.markdown(result["answer"])
        if result.get("sources"):
            st.markdown("**Sources**")
            for source in result["sources"]:
                st.markdown(f"- [{source['title']}]({source['url']})")
        with st.expander("Agent steps", expanded=True):
            for step in result["steps"]:
                st.markdown(f"**{step['title']}**")
                st.write(step["detail"])

if st.session_state.last_steps:
    with st.sidebar:
        st.subheader("Latest run")
        for step in st.session_state.last_steps:
            st.markdown(f"**{step['title']}**")
            st.caption(step["detail"])
        if st.session_state.last_sources:
            st.subheader("Latest sources")
            for source in st.session_state.last_sources:
                st.markdown(f"- [{source['title']}]({source['url']})")

# IPL 2026 AI Agent

## Objective

Create a simple, fast, and interactive AI agent that can answer user queries related **ONLY** to IPL 2026 using real-time data when required. The agent must demonstrate tool usage, decision-making, and clear response generation.

---

## Tech Requirements

- **LangGraph** - Build the agent workflow
- **Groq API** - LLM backend
- **Model** - Any fast Groq-supported model (e.g., LLaMA / Mixtral)
- **Web Search** - Integrate built-in web search feature of Groq for live web search

---

## Agent Capabilities

The agent must be able to:

### Answer Questions Like:
- "Who played yesterday in IPL 2026?"
- "What is the latest IPL 2026 points table?"
- "Who has the Orange Cap?"
- "What was the latest match result?"

### Core Requirements:
- Use live web search for time-sensitive queries
- Restrict all answers strictly to IPL 2026
- Provide short, clear, factual responses
- Avoid hallucination (if data is not found, say: "No recent IPL 2026 data found")

---

## Agent Architecture

Design a simple LangGraph workflow with the following structure:

```
User Query
    ↓
Router Node (decides whether web search is needed)
    ↓
    ├── If NOT needed → Direct LLM response
    │
    └── If needed → Web Search Tool → LLM summarizes results
                            ↓
                      Final Answer
```

---

## Routing Logic

### Use Web Search When:
- The query is about recent matches
- The query includes words like "latest", "today", "yesterday", "current"

### Direct LLM Response When:
- General knowledge questions about IPL 2026
- Static information that doesn't require real-time data

---

## System Prompt

```
You are an IPL 2026 expert AI agent.

Rules:
- Only answer questions about IPL 2026
- Always prefer the latest available data
- Use web search for real-time or uncertain queries
- Keep responses short and factual
- If no data is found, respond with 'No recent IPL 2026 data found'
- Do not make up answers
```

---

## Web Search Constraints

- Always prioritize IPL 2026 data only
- Ignore older IPL seasons unless explicitly asked
- Extract only relevant information:
  - Scores
  - Teams
  - Standings
  - Players

---

## UI Requirements

Create a simple interface (Streamlit or basic HTML) with:

### Components:
1. **Text Input** - For user queries
2. **Response Display Area** - Show agent responses
3. **Quick Demo Buttons** - Three dedicated one-click buttons

### Quick Demo Buttons:

| Button | Label | Prompt |
|--------|-------|--------|
| 1 | Points Table | "Show the latest IPL 2026 points table" |
| 2 | Orange Cap | "Who is leading the Orange Cap in IPL 2026?" |
| 3 | Latest Result | "What was the result of the latest IPL 2026 match?" |

---

## Behavior Expectations

### The agent should clearly demonstrate:
- **Decision-making** - When to use tools
- **Tool usage** - Web search integration
- **Response synthesis** - Clear, concise answers

### Performance:
- Responses should be fast (leverage Groq speed)
- Keep the system simple (single agent + one tool)

---

## Optional Enhancements

- [ ] Display reasoning steps or logs
- [ ] Add loading indicator ("Thinking...")
- [ ] Cache recent results for faster responses

---

## Final Goal

A working AI agent that:

- ✅ Reliably answers IPL 2026 questions
- ✅ Uses web search intelligently
- ✅ Demonstrates how AI agents function in a clear and simple way
- ✅ Provides an interactive experience with quick-access prompts

---

## Project Structure (Suggested)

```
ipl_2026_agent/
├── app.py              # Streamlit UI
├── agent.py            # LangGraph agent logic
├── tools.py            # Web search tool
├── config.py           # API keys and settings
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```

## Dependencies

```txt
langgraph
langchain
langchain-groq
groq
streamlit
python-dotenv
```

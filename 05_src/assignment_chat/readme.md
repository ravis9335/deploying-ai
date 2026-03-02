# ARIA — Adaptive Research and Information Assistant

ARIA is a conversational AI assistant with a curious, enthusiastic personality.
She draws on three distinct backend services to answer questions about weather,
factual knowledge, and mathematics.

---

## Chat Client Personality

ARIA presents herself as an intellectually curious and warm research companion.
She adds relevant context to answers, uses phrases like "Fascinating!" and
"Great question!", and always grounds her responses in tool output rather than
guessing. She declines certain topics (see Guardrails below) politely and
redirects the user.

---

## Services

### Service 1 — Weather API (`services/weather_api.py`)

**Backend:** [Open-Meteo](https://open-meteo.com/) — a free, open-source
weather API that requires no API key.

**How it works:**

1. The user asks about weather for a city.
2. ARIA calls `get_weather` via function calling.
3. The service first geocodes the city name using the Open-Meteo Geocoding API
   (returns latitude, longitude, and country).
4. It then fetches current conditions (temperature, feels-like, humidity, wind
   speed, precipitation, weather condition code) from the Forecast API.
5. Raw JSON is returned to the LLM, which transforms it into a natural,
   conversational weather report — never verbatim JSON.

**Example query:** *"What's the weather like in Paris right now?"*

---

### Service 2 — Semantic Knowledge Search (`services/knowledge_search.py`)

**Backend:** [ChromaDB](https://docs.trychroma.com/) with file persistence +
OpenAI `text-embedding-3-small` embeddings.

**Dataset:** `data/knowledge_base.csv` — a hand-curated CSV containing 80
entries spanning five categories: Space, History, Science, Technology, and
Geography. Each entry has an `id`, `title`, `content`, and `category`.
File size is approximately 40 KB.

**Embedding process:**

Each knowledge entry is embedded as `"{title}: {content}"` using the OpenAI
`text-embedding-3-small` model (1,536-dimension vectors). The embeddings are
stored in a ChromaDB collection with cosine similarity as the distance metric.

- On **first run**, the app checks whether the ChromaDB collection exists
  (`data/chroma_db/`). If not, it embeds all 80 entries automatically and
  saves the index. This takes roughly 30–60 seconds and uses minimal API
  credits (~80 embeddings × ~150 tokens each ≈ 12,000 tokens at ~$0.0002).
- On **subsequent runs**, the persisted index is loaded instantly.
- To pre-build the index manually, run: `python setup_embeddings.py`
- To rebuild from scratch: `python setup_embeddings.py --rebuild`

**How it works:**

1. The user asks a factual question.
2. ARIA calls `search_knowledge` via function calling with a `query` string.
3. The query is embedded using the same model, then compared against stored
   vectors using cosine similarity in ChromaDB.
4. The top-N results (default 3) are returned to the LLM.
5. ARIA synthesises a response from the relevant entries, citing categories
   and adding context.

**Example query:** *"Tell me something fascinating about black holes."*

---

### Service 3 — Calculator & Unit Converter (`services/calculator.py`)

**Mechanism:** OpenAI Function Calling (no external API required).

ARIA exposes two mathematical tools to the LLM:

- **`calculate`** — evaluates arithmetic expressions, powers, roots,
  logarithms, and trigonometric functions using a safe AST-based evaluator.
  Python's `eval()` is never used; only a whitelist of operators and functions
  is permitted, preventing code injection.
- **`convert_units`** — converts values between common units of temperature,
  distance, weight, volume, speed, area, and time.

**Why function calling?**
LLMs can make arithmetic mistakes. By routing all numerical work through
deterministic Python functions via function calling, ARIA always returns
exact results. The LLM's role is to identify *what* to calculate and then
interpret the result in natural language.

**Supported operations:**
`+`, `-`, `*`, `/`, `//`, `%`, `**`, `sqrt`, `cbrt`, `abs`, `round`,
`ceil`, `floor`, `sin`, `cos`, `tan`, `log`, `log10`, `log2`, `exp`,
`factorial`, constants `pi` and `e`.

**Supported unit categories:**
Temperature (celsius/fahrenheit/kelvin), Distance (km/miles/m/feet/inches),
Weight (kg/lbs/g/oz), Volume (l/ml/gallons), Speed (kmh/mph/ms),
Area (km²/mi²), Time (hours/minutes/days/years).

**Example queries:**
- *"Calculate 2 to the power of 32."*
- *"Convert 100 km to miles."*
- *"What is the square root of 1,764?"*

---

## User Interface

Built with [Gradio 5](https://www.gradio.app/) using `gr.ChatInterface`
with `type="messages"` for clean conversation history management.

Features:
- Example queries pre-loaded for quick exploration.
- "Undo" and "Clear conversation" buttons.
- Soft colour theme for a polished appearance.

---

## Conversation Memory

ARIA maintains the full conversation history for the duration of a session.
When the history exceeds **15 user/assistant pairs** (30 messages), a sliding
window retains the most recent 30 messages and inserts a brief context note so
the model knows earlier turns were trimmed. This prevents the context window
from overflowing on very long conversations while preserving recent context.

---

## Guardrails

### System Prompt Protection

ARIA uses regex pattern matching to detect attempts to:
- Reveal or repeat the system prompt.
- Override or ignore instructions (e.g., jailbreak phrasing).
- Adopt an alternative persona without restrictions.

When detected, ARIA politely declines and redirects.

### Restricted Topics

ARIA will not engage with questions about:
- **Cats or dogs** — detected via keyword matching on feline/canine vocabulary.
- **Horoscopes or zodiac signs** — detected via keywords for astrology
  terminology and all twelve zodiac sign names.
- **Taylor Swift** — detected via name and fan community terms.

These restrictions are enforced *before* the message reaches the LLM,
so there is no risk of the model accidentally engaging with the topic.

---

## Project Structure

```
assignment_chat/
├── app.py                  # Main Gradio application
├── setup_embeddings.py     # Optional: pre-build the ChromaDB index
├── readme.md               # This file
├── services/
│   ├── __init__.py
│   ├── weather_api.py      # Service 1: Open-Meteo weather
│   ├── knowledge_search.py # Service 2: ChromaDB semantic search
│   └── calculator.py       # Service 3: Function calling math
└── data/
    ├── knowledge_base.csv  # 80 curated knowledge entries
    └── chroma_db/          # Created automatically on first run
```

---

## Running the App

1. Ensure your `OPENAI_API_KEY` is set in a `.secrets` file at the repository
   root (or as an environment variable).

2. From the `05_src/assignment_chat/` directory:

   ```bash
   python app.py
   ```

   Or from the repository root using `uv`:

   ```bash
   uv run 05_src/assignment_chat/app.py
   ```

3. Open the URL printed by Gradio (typically `http://127.0.0.1:7860`).

> **Note:** The first time a knowledge query is made, the app will spend
> roughly 30–60 seconds building the ChromaDB index. Subsequent startups
> are immediate.

---

## Implementation Decisions

| Decision | Rationale |
|---|---|
| Open-Meteo for weather | Free, no API key, high reliability, returns structured JSON |
| `text-embedding-3-small` | Good accuracy, low cost, 1,536-dim vectors |
| ChromaDB PersistentClient | File-based, no Docker needed, as specified in the assignment |
| Cosine similarity in ChromaDB | Standard choice for embedding similarity; independent of vector magnitude |
| AST-based safe evaluator | Avoids `eval()` security risks while supporting all common math operations |
| Sliding-window memory (15 pairs) | Balances context coverage with token budget; most conversations fit entirely |
| Pre-LLM guardrail checks | Regex runs in microseconds; ensures restricted topics never reach the model |
| `gr.ChatInterface(type="messages")` | Modern Gradio 5 pattern; cleaner than manual `gr.Blocks` for this use case |

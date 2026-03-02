# ARIA — Adaptive Research and Information Assistant

ARIA is a conversational AI assistant with a curious, enthusiastic personality.
She draws on three distinct backend services to answer questions about weather,
factual knowledge, and mathematics.



## Chat Client Personality

ARIA presents herself as an intellectually curious and warm research companion.
She adds relevant context to answers, uses phrases like "Fascinating!" and
"Great question!", and always grounds her responses in tool output rather than
guessing. She declines certain topics (see Guardrails below) politely and
redirects the user.


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


### Service 2 — Semantic Knowledge Search (`services/knowledge_search.py`)

**Backend:** [ChromaDB](https://docs.trychroma.com/) with file persistence +
OpenAI `text-embedding-3-small` embeddings.

**Dataset:** `data/knowledge_base.csv` — a hand-curated CSV containing 80
entries spanning five categories: Space, History, Science, Technology, and
Geography. Each entry has an `id`, `title`, `content`, and `category`.


**How it works:**

1. The user asks a factual question.
2. ARIA calls `search_knowledge` via function calling with a `query` string.
3. The query is embedded using the same model, then compared against stored
   vectors using cosine similarity in ChromaDB.
4. The top-N results (default 3) are returned to the LLM.
5. ARIA synthesises a response from the relevant entries, citing categories
   and adding context.

**Example query:** *"Tell me something fascinating about black holes."*


### Service 3 — Calculator & Unit Converter (`services/calculator.py`)

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

**Example queries:**
- *"Calculate 2 to the power of 32."*
- *"Convert 100 km to miles."*
- *"What is the square root of 1,764?"*

---

## User Interface

Built with [Gradio 5](https://www.gradio.app/) using `gr.ChatInterface`
with `type="messages"` for clean conversation history management.




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

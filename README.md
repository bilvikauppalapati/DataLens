# DataLens 🔎

Upload, preview, clean, filter, summarize, and visualize (2D + 3D) one or
more datasets — plus discover relationships across datasets and chat with
an AI assistant (Google Gemini) about them.

## Project structure

```
datalens/
├── core/                    # Pure Python logic — NO streamlit imports here.
│   ├── loaders.py           # Phase 1 — read csv/xlsx/json/parquet, profile dataset
│   ├── preprocessing.py     # Phase 2 — cleaning, missing values, dtype conversion
│   ├── stats.py             # Phase 3 — filtering, summary statistics
│   ├── viz.py                # Phase 4/5 — 2D and 3D chart builders (Plotly)
│   ├── multi_dataset.py     # Phase 6 — side-by-side dataset comparison helpers
│   ├── relations.py         # Phase 7 — cross-dataset relationships (shared columns, joins, correlation)
│   └── chatbot.py           # Phase 8 — LLM-powered dataset Q&A (Google Gemini API)
├── app/
│   ├── main.py               # Streamlit entrypoint / home page
│   ├── session_state.py     # Manages multiple loaded datasets in memory
│   └── pages/                # One file per sidebar page (Streamlit auto-navigation)
│       ├── 1_Upload_and_Preview.py
│       ├── 2_Preprocessing.py
│       ├── 3_Filter_and_Summary.py
│       ├── 4_Visualize_2D.py
│       ├── 5_Compare_Datasets.py
│       ├── 6_Visualize_3D.py
│       ├── 7_Relationships.py
│       └── 8_Chatbot.py
├── data/sample/               # Small sample CSVs for testing
├── tests/                     # pytest tests for core/ logic
├── .streamlit/config.toml    # Theme + upload size config
├── requirements.txt
├── .env.example
└── .gitignore
```

**Design rule:** everything in `core/` is plain Python/pandas with zero
Streamlit imports. The `app/` layer only calls into `core/` and renders
results. This means the whole `core/` package can be reused later behind
a FastAPI backend without a rewrite, if this grows past an MVP.

## Setup

```bash
git clone <your-repo-url>
cd datalens
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Chatbot setup (Phase 8 — Google Gemini API)

The chatbot uses the [Google Gemini API](https://ai.google.dev/) — no local
model server needed.

1. Get an API key at <https://aistudio.google.com/app/apikey>.
2. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`:

   ```bash
   cp .env.example .env
   # then edit .env: GEMINI_API_KEY=your-key
   ```

   Alternatively, export `GEMINI_API_KEY` in your shell or put it in
   `.streamlit/secrets.toml` (both are git-ignored).
3. Optional: set `GEMINI_MODEL` to use a different model (default
   `gemini-3.5-flash`).

Only the schema, summary stats and a few sample rows are sent to Gemini —
never the full dataset.

## Run

```bash
streamlit run app/main.py
```

## Run tests

```bash
pip install pytest
pytest tests/ -v
```

## Build phases

- [x] **Phase 0** — Project structure & architecture
- [x] **Phase 1** — Upload & Preview (CSV/Excel/JSON/Parquet, dataset profiling)
- [x] **Phase 2** — Preprocessing (missing values, type conversion — str/int/float/decimal only, duplicates, outliers)
- [x] **Phase 3** — Filtering & summary statistics
- [x] **Phase 4** — 2D visualization
- [x] **Phase 6** — Multi-dataset support (side-by-side view)
- [x] **Phase 5** — 3D visualization (scatter + surface)
- [x] **Phase 7** — Cross-dataset relationship discovery (shared columns, join key suggestions, cross-dataset correlation)
- [ ] **Phase 8** — Chatbot UI (Gemini-backed, in `core/chatbot.py` + `app/pages/8_Chatbot.py`)
- [ ] **Phase 9** — Polish, auth, persistence, deployment

## Sample data

`data/sample/` has three small CSVs for manual testing:
- `iris_sample.csv` — classic numeric dataset, good for 2D/3D charts
- `countries_sample.csv` — country/population/gdp/continent
- `country_happiness_sample.csv` — shares `country` values with the file
  above, so you can test the **Relationships** page (join + cross-dataset
  correlation) with real overlapping data

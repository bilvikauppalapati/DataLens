import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from core.chatbot import check_ollama_available, build_context, ask_chatbot
from core.lookup import find_matching_rows, cross_dataset_lookup
from core.nlq import detect_lookup_rule_based, resolve_structured_query
from core.sql_query import answer_with_sql
from core.loaders import profile_dataset
from core.stats import summary_stats, apply_filters
from app.session_state import init_session_state, list_dataset_names

st.set_page_config(page_title="DataLens — Chatbot", page_icon="💬", layout="wide")
init_session_state()

st.title("💬 Chatbot")
st.caption(
    "Ask about your dataset in plain English. Exact value lookups (e.g. "
    "\"what is X where Y = Z\", including across two datasets) always run as "
    "real, exact data lookups — never a guess from the AI."
)

dataset_names = list_dataset_names()
if not dataset_names:
    st.info("No datasets loaded yet. Go to **Upload & Preview** first.")
    st.stop()

if not check_ollama_available():
    st.error(
        "Can't reach Ollama at http://localhost:11434. Make sure it's installed and running:\n\n"
        "```\nollama serve\n```\n\nand that you've pulled a model, e.g. `ollama pull llama3.2:3b`."
    )
    st.stop()

all_dfs = {name: st.session_state.datasets[name]["df"] for name in dataset_names}
dataset_columns = {name: list(df.columns) for name, df in all_dfs.items()}

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

st.divider()

# ---------- Manual lookup (guaranteed reliable, no LLM parsing needed) ----------
with st.expander("🔍 Quick Lookup (manual — always exact, skips the chatbot)"):
    lookup_scope = st.radio("Lookup type", ["Within one dataset", "Across two datasets"], horizontal=True)

    if lookup_scope == "Within one dataset":
        ds = st.selectbox("Dataset", options=dataset_names, key="ql_ds")
        cols = dataset_columns[ds]
        fcol = st.selectbox("Filter column", options=cols, key="ql_fcol")
        fval = st.text_input("Filter value", key="ql_fval")
        if st.button("Look up", key="ql_btn"):
            if fval:
                try:
                    result = find_matching_rows(all_dfs[ds], fcol, fval)
                    if result.empty:
                        st.warning(f"No rows found where '{fcol}' = '{fval}'.")
                    else:
                        st.dataframe(result, use_container_width=True)
                except ValueError as e:
                    st.error(str(e))
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            ds_a = st.selectbox("First dataset (has the value you know)", options=dataset_names, key="ql_ds_a")
        with col_b:
            ds_b_options = [n for n in dataset_names if n != ds_a]
            ds_b = st.selectbox("Second dataset (look up the answer here)", options=ds_b_options, key="ql_ds_b")

        shared = sorted(set(dataset_columns[ds_a]) & set(dataset_columns[ds_b]))
        if not shared:
            st.warning(f"'{ds_a}' and '{ds_b}' don't share any column name to join on.")
        else:
            join_col = st.selectbox("Shared column to match on", options=shared, key="ql_join")
            join_val = st.text_input(f"Value of '{join_col}' to look up", key="ql_join_val")
            if st.button("Look up", key="ql_btn_cross"):
                if join_val:
                    try:
                        result = cross_dataset_lookup(all_dfs[ds_a], all_dfs[ds_b], join_col, join_val)
                        st.dataframe(result, use_container_width=True)
                    except ValueError as e:
                        st.error(str(e))

st.divider()

# ---------- Chat ----------
selected_ds_for_context = st.selectbox(
    "Dataset for general questions (used when you're not doing a cross-dataset lookup)",
    options=dataset_names,
)

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("table") is not None:
            st.dataframe(msg["table"], use_container_width=True)

question = st.chat_input("Ask about your dataset...")

if question:
    st.session_state.chat_messages.append({"role": "user", "content": question, "table": None})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = None
            table = None

            # 1. Range / multi-condition / distinct-values queries ("between X
            # and Y", "bmw and tesla with 4 doors", "list all fuel types",
            # "total 6 car brand names") — see core.nlq.resolve_structured_query
            # for why the priority between these two detectors is load-bearing
            # (a generic trigger word like "list" must never win over an
            # actual, data-matched filter condition).
            structured_query = resolve_structured_query(question, all_dfs[selected_ds_for_context])

            if structured_query and structured_query["kind"] == "distinct":
                col = structured_query["column"]
                values = structured_query["values"]
                count = structured_query["count"]
                shown = ", ".join(str(v) for v in values[:50])
                answer = (
                    f"'{col}' in '{selected_ds_for_context}' has {count} distinct value(s): {shown}"
                    + (" ..." if count > 50 else "")
                )
                table = pd.DataFrame({col: values})

            elif structured_query and structured_query["kind"] == "filter":
                try:
                    result = apply_filters(all_dfs[selected_ds_for_context], structured_query["filters"])
                    filter_desc = ", ".join(
                        f"{col} {spec}" for col, spec in structured_query["filters"].items()
                    )
                    if result.empty:
                        answer = f"No rows in '{selected_ds_for_context}' match: {filter_desc}."
                    else:
                        answer = (
                            f"Found {len(result)} matching row(s) in '{selected_ds_for_context}' "
                            f"where {filter_desc}."
                        )
                        table = result.head(200)
                        if len(result) > 200:
                            answer += " Showing the first 200."
                except ValueError as e:
                    answer = str(e)

            else:
                # 3. Try a fast, deterministic (non-LLM) single-value lookup first —
                # this is what actually fixes the "not enough information" bug: it
                # searches the REAL data directly instead of depending on the small
                # local model formatting JSON correctly every time.
                intent = detect_lookup_rule_based(question, all_dfs)

                if intent and intent.get("intent") == "lookup":
                    try:
                        if intent["scope"] == "single":
                            df = all_dfs[intent["dataset"]]
                            result = find_matching_rows(df, intent["filter_column"], intent["filter_value"])
                            if result.empty:
                                answer = (
                                    f"No rows found in '{intent['dataset']}' where "
                                    f"'{intent['filter_column']}' = '{intent['filter_value']}'."
                                )
                            else:
                                values = result[intent["target_column"]].tolist()
                                answer = (
                                    f"Found it — {intent['target_column']} = {values[0]!r} "
                                    f"(from '{intent['dataset']}' where {intent['filter_column']} = "
                                    f"{intent['filter_value']!r})."
                                )
                                table = result
                        else:  # cross
                            df_a = all_dfs[intent["dataset_a"]]
                            df_b = all_dfs[intent["dataset_b"]]
                            result = cross_dataset_lookup(
                                df_a, df_b, intent["join_column"], intent["filter_value"]
                            )
                            values = result[intent["target_column"]].tolist()
                            answer = (
                                f"Found it — {intent['target_column']} = {values[0]!r} in "
                                f"'{intent['dataset_b']}', matched via {intent['join_column']} = "
                                f"{intent['filter_value']!r} from '{intent['dataset_a']}'."
                            )
                            table = result

                    except ValueError as e:
                        answer = str(e)

                else:
                    # 4. Text-to-SQL: for everything the rule-based detectors above
                    # don't cover — aggregates (SUM/COUNT/AVG), GROUP BY, multi-table
                    # JOINs, or any other phrasing — ask the model to translate the
                    # question into a real SQL query and RUN it against the actual
                    # data (see core.sql_query), instead of asking it to fit the
                    # answer into a fixed "one column equals one value" JSON shape.
                    # This supersedes the old detect_lookup_intent classifier, which
                    # could only ever express what its schema anticipated.
                    sql_result = answer_with_sql(question, all_dfs)

                    if sql_result:
                        sql_used, result = sql_result
                        if result.empty:
                            answer = f"Ran this query but it returned no rows:\n```sql\n{sql_used}\n```"
                        else:
                            answer = f"```sql\n{sql_used}\n```"
                            table = result

                    else:
                        # 5. Last resort: a genuinely open-ended question (e.g. a
                        # summary) that isn't a data-retrieval query at all, or the
                        # model couldn't produce a valid query — fall back to a
                        # free-text answer grounded in a lightweight data profile.
                        df = all_dfs[selected_ds_for_context]
                        data = st.session_state.datasets[selected_ds_for_context]
                        profile = profile_dataset(df)
                        try:
                            stats = summary_stats(df).reset_index().to_dict(orient="records")
                        except Exception:
                            stats = None
                        sample_rows = df.head(5).to_dict(orient="records")
                        context = build_context(profile, stats, sample_rows, data["transform_history"])

                        try:
                            answer = ask_chatbot(question, context)
                        except (ConnectionError, RuntimeError) as e:
                            answer = str(e)

            if answer is not None:
                st.write(answer)
            if table is not None:
                st.dataframe(table, use_container_width=True)

    st.session_state.chat_messages.append({"role": "assistant", "content": answer, "table": table})

if st.session_state.chat_messages:
    if st.button("Clear chat"):
        st.session_state.chat_messages = []
        st.rerun()
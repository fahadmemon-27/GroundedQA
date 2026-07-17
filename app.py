import streamlit as st
import json
import pandas as pd
import altair as alt
from llm_baseline import generate_baseline_answer
from rag_pipeline import generate_rag_answer, retrieve_chunks
from rad_pipeline import generate_rad_answer
from nli_scorer import score_answer

st.set_page_config(page_title="Hallucination Detection: LLM vs RAG vs RAD", layout="wide")

st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 20% 20%, #1a1a1a 0%, #0a0a0a 60%);
        color: #e8e8e8;
    }
    h1, h2, h3 {
        color: #f5f5f5;
        font-weight: 700;
    }
    p, label {
        color: #b0b0b0 !important;
    }
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.06);
        color: #f5f5f5;
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 0.85rem 1.1rem;
        font-size: 1rem;
        outline: none !important;
    }
    .stTextInput > div > div > input:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    .stButton > button {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        color: #f0f0f0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 2rem;
    }
    .stButton > button:hover {
        background: rgba(255, 255, 255, 0.15);
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 1.5rem;
        min-height: 220px;
    }
    .glass-card h3 {
        color: #f5f5f5;
        margin-top: 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.15);
        padding-bottom: 0.5rem;
    }
    .glass-card p.answer-text {
        color: #d8d8d8;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .score-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 800;
        font-size: 0.85rem;
        margin-bottom: 0.8rem;
    }
    .score-high { background: #2ecc71; color: #0a0a0a; }
    .score-mid { background: #f1c40f; color: #0a0a0a; }
    .score-low { background: #e74c3c; color: #0a0a0a; }

    div[data-testid="stTabs"] [role="tablist"] {
        justify-content: flex-end;
        gap: 10px;
        border-bottom: none !important;
        box-shadow: none !important;
    }
    div[data-testid="stTab"] {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        color: #d0d0d0 !important;
        font-weight: 600;
        box-shadow: none !important;
    }
    div[data-testid="stTab"]:hover {
        background: rgba(255, 255, 255, 0.15) !important;
        color: #ffffff !important;
    }
    div[data-testid="stTab"][aria-selected="true"] {
        background: rgba(255, 255, 255, 0.2) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.35) !important;
    }
</style>
""", unsafe_allow_html=True)


def render_panel(title, answer, score, color_class):
    html = f"""
    <div class="glass-card">
        <h3>{title}</h3>
        <span class="score-badge {color_class}">Groundedness: {score:.0f}%</span>
        <p class="answer-text">{answer}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def make_chart(pipeline_names, values):
    PIPELINE_COLORS = {"Baseline": "#2ecc71", "RAG": "#f1c40f", "RAD": "#e74c3c"}
    chart_data = pd.DataFrame({"Pipeline": pipeline_names, "Groundedness": values})
    chart_data["Color"] = chart_data["Pipeline"].map(PIPELINE_COLORS)

    chart = alt.Chart(chart_data).mark_bar(size=70, cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
        x=alt.X("Pipeline", sort=None, axis=alt.Axis(labelColor="#e8e8e8", titleColor="#e8e8e8")),
        y=alt.Y("Groundedness", axis=alt.Axis(labelColor="#e8e8e8", titleColor="#e8e8e8")),
        color=alt.Color("Color", scale=None, legend=None)
    ).properties(height=300, background="transparent").configure_view(strokeWidth=0)

    return chart


# Initialize session state for live tracking
if "live_results" not in st.session_state:
    st.session_state["live_results"] = []


tab1, tab2 = st.tabs(["Home", "Evaluation Dashboard"])

with tab1:
    st.title("Hallucination Detection: LLM vs RAG vs RAD")
    st.write("Ask a question and compare answers from three approaches, along with their groundedness scores.")

    question = st.text_input("Enter your question:", key="question_input")
    submit_clicked = st.button("Submit")

    if submit_clicked or (question and st.session_state.get("last_question") != question):
        st.session_state["last_question"] = question

        with st.spinner("Retrieving evidence and generating answers... this may take a minute (RAD is slower)."):
            evidence_chunks = retrieve_chunks(question)

            baseline_answer = generate_baseline_answer(question)
            baseline_score = score_answer(baseline_answer, evidence_chunks)

            rag_answer, _ = generate_rag_answer(question)
            rag_score = score_answer(rag_answer, evidence_chunks)

            rad_answer, _ = generate_rad_answer(question)
            rad_score = score_answer(rad_answer, evidence_chunks)

        # Save this result into the live session tracker for the Dashboard
        st.session_state["live_results"].append({
            "question": question,
            "baseline": {"answer": baseline_answer, "groundedness": baseline_score["groundedness_score"]},
            "rag": {"answer": rag_answer, "groundedness": rag_score["groundedness_score"]},
            "rad": {"answer": rad_answer, "groundedness": rad_score["groundedness_score"]},
        })

        col1, col2, col3 = st.columns(3)
        with col1:
            render_panel("LLM Baseline", baseline_answer, baseline_score["groundedness_score"], "score-high")
        with col2:
            render_panel("RAG", rag_answer, rag_score["groundedness_score"], "score-mid")
        with col3:
            render_panel("RAD", rad_answer, rad_score["groundedness_score"], "score-low")

with tab2:
    st.title("Evaluation Dashboard")

    # --- Small benchmark reference section at the top ---
    st.subheader("Benchmark Reference (fixed test set)")
    try:
        with open("evaluation_results.json", "r", encoding="utf-8") as f:
            benchmark_results = json.load(f)

        bench_baseline = sum(r["baseline"]["groundedness"] for r in benchmark_results) / len(benchmark_results)
        bench_rag = sum(r["rag"]["groundedness"] for r in benchmark_results) / len(benchmark_results)
        bench_rad = sum(r["rad"]["groundedness"] for r in benchmark_results) / len(benchmark_results)

        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            st.metric("Baseline (benchmark)", f"{bench_baseline:.1f}%")
        with bc2:
            st.metric("RAG (benchmark)", f"{bench_rag:.1f}%")
        with bc3:
            st.metric("RAD (benchmark)", f"{bench_rad:.1f}%")

    except FileNotFoundError:
        st.info("No benchmark file found (evaluation_results.json).")

    st.divider()

    # --- Live, session-based results ---
    st.subheader("Live Results (this session)")

    live = st.session_state["live_results"]

    if len(live) == 0:
        st.info("No questions asked yet this session. Go to the Home tab and ask something!")
    else:
        live_baseline = sum(r["baseline"]["groundedness"] for r in live) / len(live)
        live_rag = sum(r["rag"]["groundedness"] for r in live) / len(live)
        live_rad = sum(r["rad"]["groundedness"] for r in live) / len(live)

        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            st.metric("Baseline (live avg)", f"{live_baseline:.1f}%")
        with lc2:
            st.metric("RAG (live avg)", f"{live_rag:.1f}%")
        with lc3:
            st.metric("RAD (live avg)", f"{live_rad:.1f}%")

        st.write(f"Based on {len(live)} question(s) asked this session.")

        chart = make_chart(["Baseline", "RAG", "RAD"], [live_baseline, live_rag, live_rad])
        st.altair_chart(chart, use_container_width=True)

        with st.expander("View questions asked this session"):
            for r in live:
                st.write(f"**{r['question']}**")
                st.write(f"Baseline: {r['baseline']['groundedness']:.0f}% | RAG: {r['rag']['groundedness']:.0f}% | RAD: {r['rad']['groundedness']:.0f}%")
                st.write("---")
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import render

st.set_page_config(page_title="lineup · case explorer", layout="wide")

st.title("lineup")
st.caption("Which retrieved chunk caused a wrong RAG answer — and which attribution methods get fooled by the near-miss.")

default_directory = str(Path(__file__).parent / "sample")
directory = st.sidebar.text_input("data directory", value=default_directory)
st.sidebar.caption("Point this at a pipeline run's outputs/ to explore real results. The bundled sample is illustrative.")

scenarios, generations, roles, predictions, predictions_by_qid = render.load(directory)

if not scenarios:
    st.warning("No data found. Generate the bundled sample with `python app/make_sample.py`, or point the sidebar at a run's outputs/ directory.")
    st.stop()

overview_tab, explorer_tab = st.tabs(["Overview", "Case explorer"])

with overview_tab:
    reports = render.scoring_reports(roles, predictions)

    st.subheader("How each method does")
    st.caption("Scored on the wrong answers — where there is an error to attribute. `misleading-as-culprit` is the headline: how often a method blames a chunk that looks guilty but is not the cause.")
    st.dataframe(render.scoring_table(reports), use_container_width=True, hide_index=True)

    st.subheader("Where the blame lands")
    st.caption("The true role of the chunk each method picked. Red is the trap — blaming a misleading near-miss.")
    st.markdown(render.role_distribution_html(reports), unsafe_allow_html=True)

    st.subheader("Does it matter — selective answering")
    st.caption("Can a confidence signal tell correct answers from wrong ones? An AUROC of one half is chance; the oracle bounds what perfect causal attribution could do.")
    st.dataframe(render.abstention_table(generations, roles, predictions), use_container_width=True, hide_index=True)

with explorer_tab:
    only_fooled = st.checkbox("only cases where a method blamed a near-miss", value=False)
    options = render.case_options(scenarios, roles, predictions_by_qid, only_fooled=only_fooled)
    if not options:
        st.info("No cases match the filter.")
    else:
        qid = st.selectbox("case", options, format_func=lambda q: render.case_label(q, scenarios, generations))
        st.markdown(render.case_detail(qid, scenarios, generations, roles, predictions_by_qid), unsafe_allow_html=True)

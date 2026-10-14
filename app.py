"""GradeSense AI: AI-powered grade change intelligence for paper manufacturing.
Run with:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="GradeSense AI", page_icon="📈", layout="wide")

from gradesense import ui                                   # noqa: E402
from gradesense.data import generate_dataset, prepare       # noqa: E402
from gradesense.model import train_model                    # noqa: E402
from gradesense.views import PAGES, build_context, install  # noqa: E402

ui.inject_css()


@st.cache_resource(show_spinner="Simulating plant data and training the model...")
def load_default():
    df, report = prepare(generate_dataset(248, seed=7))
    return df, report, train_model(df)


S = st.session_state
if "bundle" not in S:
    S.update(limit=2.5, max_steam=6, max_speed=6, max_stock=4, feedback=[], scenarios=[], chat=[])
    install(S, *load_default())

with st.sidebar:
    st.markdown('<div class="gs-brand">GradeSense AI</div><div class="gs-brand-sub">AI-powered grade change intelligence</div>',
                unsafe_allow_html=True)
    page = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.markdown("**Live scenario**")
    ids = sorted(S["bundle"].test_ids)
    S["tid"] = st.selectbox("Grade change", ids, index=ids.index(S["tid"]) if S["tid"] in ids else 0,
                            format_func=lambda i: f"#{i}", key="sel_tid")
    n_min = int(S["df"][S["df"].transition_id == S["tid"]].t.max())
    S["minute"] = min(S["minute"], n_min)
    st.slider("Minute in transition", 0, n_min, key="minute")
    b1, b2 = st.columns(2)
    b1.button("+1 min", on_click=lambda: S.update(minute=min(S["minute"] + 1, n_min)))
    b2.button("+5 min", on_click=lambda: S.update(minute=min(S["minute"] + 5, n_min)))
    st.caption("Replays a held-out grade change the model never saw during training.")

PAGES[page](build_context(S), S)

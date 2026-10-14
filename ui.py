"""Theme, HTML helpers and Plotly styling shared by all pages."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

C = dict(bg="#0A1020", panel="#0F1830", border="#1F2D4D", text="#E6EDF7", muted="#8A9BBE",
         blue="#3B82F6", green="#22C55E", amber="#F5A524", red="#F04444", cyan="#38BDF8")
TONE = {"ok": C["green"], "warn": C["amber"], "bad": C["red"], "info": C["blue"], "neutral": C["text"]}
RISK_TONE = {"LOW": "ok", "MEDIUM": "warn", "HIGH": "bad"}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
.stMarkdown, .gs-kpi, .gs-title, .gs-banner, .gs-kv, .gs-act {{ font-family: 'IBM Plex Sans', sans-serif; }}
.block-container {{ padding-top: 1.2rem; max-width: 1560px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
section[data-testid="stSidebar"] {{ background: #080D1B; border-right: 1px solid {C['border']}; }}
.gs-brand {{ font-size: 1.25rem; font-weight: 600; margin: 0 0 .1rem 0; }}
.gs-brand-sub {{ color: {C['muted']}; font-size: .78rem; margin-bottom: .8rem; }}
.gs-head {{ display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:.8rem; gap:1rem; flex-wrap:wrap; }}
.gs-head h1 {{ font-size: 1.55rem; font-weight: 600; margin:0; padding:0; }}
.gs-head p {{ color:{C['muted']}; margin:.15rem 0 0 0; font-size:.88rem; }}
.gs-pill {{ display:inline-block; border:1px solid {C['border']}; border-radius:999px; padding:.18rem .7rem; font-size:.78rem; margin-left:.4rem; background:{C['panel']}; }}
.gs-dot {{ display:inline-block; width:.5rem; height:.5rem; border-radius:50%; margin-right:.4rem; }}
.gs-kpi {{ background:{C['panel']}; border:1px solid {C['border']}; border-radius:8px; padding:.7rem .9rem; height:100%; }}
.gs-kpi-l {{ color:{C['muted']}; font-size:.78rem; }}
.gs-kpi-v {{ font-size:1.7rem; font-weight:600; font-variant-numeric: tabular-nums; line-height:1.25; }}
.gs-kpi-s {{ color:{C['muted']}; font-size:.76rem; }}
.gs-title {{ font-weight:600; font-size:.95rem; margin:.1rem 0 .1rem 0; }}
.gs-sub {{ color:{C['muted']}; font-size:.78rem; margin-bottom:.3rem; }}
.gs-banner {{ border-radius:6px; padding:.55rem .9rem; margin:.7rem 0; font-size:.88rem; border:1px solid; }}
.gs-kv {{ display:flex; justify-content:space-between; padding:.38rem 0; border-bottom:1px solid {C['border']}; font-size:.86rem; }}
.gs-kv span:first-child {{ color:{C['muted']}; }}
.gs-act {{ display:flex; justify-content:space-between; padding:.5rem 0; border-bottom:1px solid {C['border']}; font-size:.9rem; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 8px; }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def kpi(label, value, sub="", tone="neutral"):
    return (f'<div class="gs-kpi"><div class="gs-kpi-l">{label}</div>'
            f'<div class="gs-kpi-v" style="color:{TONE[tone]}">{value}</div>'
            f'<div class="gs-kpi-s">{sub}</div></div>')


def kpi_row(items):
    for col, it in zip(st.columns(len(items)), items):
        col.markdown(kpi(*it), unsafe_allow_html=True)


def header(title, subtitle="", pills=()):
    p = "".join(f'<span class="gs-pill"><span class="gs-dot" style="background:{TONE[t]}"></span>{x}</span>'
                for x, t in pills)
    st.markdown(f'<div class="gs-head"><div><h1>{title}</h1><p>{subtitle}</p></div><div>{p}</div></div>',
                unsafe_allow_html=True)


def banner(text, tone="bad"):
    col = TONE[tone]
    st.markdown(f'<div class="gs-banner" style="color:{col};border-color:{col}55;background:{col}14">{text}</div>',
                unsafe_allow_html=True)


def title(text, sub=""):
    st.markdown(f'<div class="gs-title">{text}</div><div class="gs-sub">{sub}</div>', unsafe_allow_html=True)


def kv(rows):
    st.markdown("".join(f'<div class="gs-kv"><span>{k}</span><span style="color:{TONE[t]}">{v}</span></div>'
                        for k, v, t in rows), unsafe_allow_html=True)


def style(fig, height=300):
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10), font=dict(color=C["text"], size=12),
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)))
    fig.update_xaxes(gridcolor="#18233F", zerolinecolor="#18233F")
    fig.update_yaxes(gridcolor="#18233F", zerolinecolor="#18233F")
    return fig


def show(fig, **kw):
    try:
        st.plotly_chart(fig, width="stretch", **kw)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, **kw)


def show_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def limit_lines(fig, limit, row=None, col=None):
    kw = dict(row=row, col=col) if row else {}
    for y in (limit, -limit):
        fig.add_hline(y=y, line=dict(color=C["amber"], dash="dash", width=1), **kw)

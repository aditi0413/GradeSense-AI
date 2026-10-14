"""Streamlit pages. Each page is a function taking the live Context `c`."""
from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression

from . import copilot, ui
from .data import HORIZON, REQUIRED_COLS, WIDTH_M, generate_dataset, prepare, transition_summary
from .model import (FEATURES, compute_features, est_steady_minutes, evaluate, group_shap,
                    offspec_prob, predict_changes, recommend, risk_label, shap_values,
                    single_effects, train_model)
from .ui import C, RISK_TONE

FEEDBACK_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "feedback_log.csv")
UNITS = {"steam": ("Steam pressure", "bar"), "speed": ("Machine speed", "m/min"), "stock": ("Stock flow", "m³/h"),
         "filler": ("Filler flow", "L/min")}


# ---------------------------------------------------------------- context
@dataclass
class Context:
    df: pd.DataFrame
    bundle: object
    summary: pd.DataFrame
    limit: float
    tid: int
    t: int
    trans: pd.DataFrame
    row: pd.Series
    trans_pred: np.ndarray
    pred: float
    p_off: float
    risk: str
    conf: float
    shap_g: pd.Series
    shap_f: pd.Series
    base: float
    rec: pd.DataFrame
    best: pd.Series
    est: float


def build_context(S) -> Context:
    df, bundle, limit = S["df"], S["bundle"], S["limit"]
    trans = df[df.transition_id == S["tid"]].reset_index(drop=True)
    t = int(min(S["minute"], len(trans) - 1))
    row = trans.iloc[t]
    F = compute_features(trans)
    preds = bundle.model.predict(F)
    pred = float(preds[t])
    p = float(offspec_prob(pred, bundle.sigma, limit))
    sv, base = shap_values(bundle, F.iloc[[t]])
    rec = recommend(bundle, row, limit, S["max_steam"], S["max_speed"], S["max_stock"])
    return Context(df, bundle, S["summary"], limit, int(S["tid"]), t, trans, row, preds, pred, p,
                   risk_label(p), max(p, 1 - p), group_shap(sv).iloc[0], sv.iloc[0], base, rec,
                   rec.iloc[0], est_steady_minutes(trans, t, pred))


def actions(best):
    names = {"steam_pct": "Steam pressure", "speed_pct": "Machine speed", "stock_pct": "Stock flow"}
    out = [(n, float(best[k])) for k, n in names.items() if abs(best[k]) > 1e-9]
    return sorted(out, key=lambda x: -abs(x[1]))


def _fmt_pct(v):
    return f"{v:+.0f}%"


def _arrow_tone(v):  # lowering is shown red/down, raising green/up (as in the deck)
    return "bad" if v < 0 else "ok"


def timeline_fig(c: Context, height=300):
    tr, t = c.trans, c.t
    fig = go.Figure()
    fig.add_hrect(y0=-c.limit, y1=c.limit, fillcolor=C["green"], opacity=.06, line_width=0)
    fig.add_trace(go.Scatter(x=tr.t[:t + 1], y=tr.dev_pct[:t + 1], name="Actual deviation %",
                             line=dict(color=C["green"], width=2)))
    fig.add_trace(go.Scatter(x=tr.t[:t + 1] + HORIZON, y=c.trans_pred[:t + 1], name=f"Forecast (+{HORIZON} min)",
                             line=dict(color=C["red"], width=2, dash="dot")))
    ui.limit_lines(fig, c.limit)
    fig.add_vline(x=t, line=dict(color=C["muted"], width=1))
    fig.update_layout(xaxis_title="Minutes since grade change started", yaxis_title="Deviation (%)")
    return ui.style(fig, height)


def gauge(value, title, color, lo=0, hi=100, suffix="%"):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value, number=dict(suffix=suffix, font=dict(size=28)),
                                 title=dict(text=title, font=dict(size=13)),
                                 gauge=dict(axis=dict(range=[lo, hi]), bar=dict(color=color),
                                            bgcolor="#18233F", borderwidth=0)))
    return ui.style(fig, 190)


def page_header(c, title, sub):
    live = ("Live", "ok")
    ui.header(title, sub, [live, (f"Transition #{c.tid}: {c.trans.grade_from[0]} → {c.trans.grade_to[0]} GSM", "info"),
                           (f"Minute {c.t}", "neutral")])


# ---------------------------------------------------------------- pages
def dashboard(c: Context, S):
    page_header(c, "Real-time grade monitoring", "Early warning for basis weight deviation during grade changes")
    dev, g1 = c.row.dev_pct, c.trans.grade_to[0]
    off = abs(dev) > c.limit
    ui.kpi_row([
        ("Current grade", f"{c.row.plan:.1f} GSM", f"Planned basis weight now", "neutral"),
        ("Target grade", f"{g1:.1f} GSM", f"Changing from {c.trans.grade_from[0]} GSM", "neutral"),
        ("Deviation now", f"{dev:+.2f}%", "Off-spec" if off else "Within spec", "bad" if off else "ok"),
        (f"Forecast +{HORIZON} min", f"{c.pred:+.2f}%", f"Limit ±{c.limit:g}%", "bad" if abs(c.pred) > c.limit else "ok"),
        ("Risk level", c.risk, f"{c.p_off:.0%} chance of leaving ±{c.limit:g}%", RISK_TONE[c.risk]),
        ("Model confidence", f"{c.conf:.0%}", "Certainty of the in/out-of-spec call", "info"),
    ])
    if abs(c.pred) > c.limit or off:
        ui.banner(f"Deviation outside the ±{c.limit:g}% quality threshold (forecast {c.pred:+.2f}%). Action recommended.", "bad")
    elif c.risk == "MEDIUM":
        ui.banner("Forecast is approaching the quality threshold. Monitor closely.", "warn")
    else:
        ui.banner("Process within spec. No action needed.", "ok")

    a, b, d, e = st.columns([3, 1.5, 2, 2])
    with a, st.container(border=True):
        ui.title("Deviation trend", "Basis weight deviation, actual vs forecast")
        ui.show(timeline_fig(c, 290))
    with b, st.container(border=True):
        ui.title("Key status")
        ui.kv([("Deviation", f"{dev:+.2f}%", "bad" if off else "ok"),
               ("Time in transition", f"{c.t} min", "neutral"),
               ("Time to steady state (est.)", f"{c.est:.1f} min", "neutral"),
               ("Confidence", f"{c.conf:.0%}", "info"),
               ("Data quality", "Good", "ok")])
    with d, st.container(border=True):
        ui.title("Top influencing variables", "Local SHAP, % of forecast deviation")
        s = c.shap_g.drop("Grade schedule")
        s = s.reindex(s.abs().sort_values().index)[-7:]
        fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                               marker_color=[C["red"] if v > 0 else C["blue"] for v in s.values]))
        fig.update_layout(xaxis_title="Effect on forecast (%)")
        ui.show(ui.style(fig, 290))
    with e, st.container(border=True):
        ui.title("Recommended action", "Top moves from the recommendation engine")
        acts = actions(c.best)
        if not acts:
            st.markdown("Hold current setpoints.")
        for i, (n, v) in enumerate(acts, 1):
            st.markdown(f'<div class="gs-act"><span>{i}. {n}</span><span style="color:{ui.TONE[_arrow_tone(v)]}">{_fmt_pct(v)}</span></div>',
                        unsafe_allow_html=True)
        st.markdown(f"**Expected deviation:** {c.pred:+.2f}% → {c.best.pred:+.2f}%")
        st.caption(f"Target: within ±{c.limit:g}%")

    a, b, d, e = st.columns([1.6, 1.8, 2, 1.6])
    with a, st.container(border=True):
        ui.title("Threshold info")
        ui.kv([("Spec limit", f"±{c.limit:g}%", "neutral"), ("Current deviation", f"{dev:+.2f}%", "bad" if off else "ok"),
               ("Status", "Action required" if off else "In spec", "bad" if off else "ok")])
        st.caption("Basis weight must stay within the limit during the whole grade change.")
    with b, st.container(border=True):
        ui.title("Historical transitions", f"{len(c.summary)} grade changes")
        n_off = int(c.summary.off_spec.sum())
        fig = go.Figure(go.Pie(labels=["Off-spec", "Within spec"], values=[n_off, len(c.summary) - n_off], hole=.62,
                               marker_colors=[C["red"], C["green"]], textinfo="percent", sort=False))
        ui.show(ui.style(fig, 200))
    with d, st.container(border=True):
        ui.title("Next recommended action")
        if acts:
            n, v = acts[0]
            st.success(f"{n}: {_fmt_pct(v)}")
        else:
            st.success("No adjustment needed")
        st.markdown(f"Predicted deviation **{c.pred:+.2f}% → {c.best.pred:+.2f}%**")
        st.markdown(f"Off-spec probability **{c.p_off:.0%} → {c.best.p_off:.0%}**")
    with e, st.container(border=True):
        ui.title("Model confidence")
        ui.show(gauge(c.conf * 100, "", C["green"]))


def analytics(c: Context, S):
    ui.header("Production analytics", "Grade-change performance across all recorded transitions", [("Historical data", "info")])
    s, df, lim = c.summary, c.df, c.limit
    off_min = (df.dev_pct.abs() > lim).mean()
    ui.kpi_row([
        ("Grade changes", f"{len(s)}", "Recorded transitions", "neutral"),
        ("Off-spec transitions", f"{s.off_spec.mean():.0%}", f"More than 3 min outside ±{lim:g}%", "bad"),
        ("Off-spec time", f"{off_min:.1%}", "Share of all minutes", "warn"),
        ("Avg stabilisation", f"{s.stabilisation_min.mean():.1f} min", "Until last off-spec minute", "info"),
        ("Paper wasted", f"{s.waste_tons.sum():,.0f} t", f"Off-spec output at {WIDTH_M:g} m width", "bad"),
    ])
    st.write("")
    a, b = st.columns(2)
    with a, st.container(border=True):
        ui.title("Basis weight deviation distribution")
        fig = go.Figure(go.Histogram(x=df.dev_pct, nbinsx=60, marker_color=C["blue"]))
        for x in (-lim, lim):
            fig.add_vline(x=x, line=dict(color=C["amber"], dash="dash"))
        fig.update_layout(xaxis_title="Deviation (%)", yaxis_title="Minutes")
        ui.show(ui.style(fig, 300))
    with b, st.container(border=True):
        ui.title("Risk correlation", f"Chance the +{HORIZON} min value is off-spec, given today's deviation")
        lab = df[df.y.notna()]
        x = lab.dev_pct.abs().values.reshape(-1, 1)
        yb = (lab.y.abs() > lim).astype(int)
        lr = LogisticRegression().fit(x, yb)
        xs = np.linspace(0, float(x.max()), 120)
        samp = lab.sample(min(1500, len(lab)), random_state=1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=samp.dev_pct.abs(), y=(samp.y.abs() > lim).astype(int) * 100, mode="markers",
                                 marker=dict(color=C["blue"], size=4, opacity=.25), name="Observations"))
        fig.add_trace(go.Scatter(x=xs, y=lr.predict_proba(xs.reshape(-1, 1))[:, 1] * 100, name="Logistic fit",
                                 line=dict(color=C["red"], width=3)))
        fig.add_vline(x=lim, line=dict(color=C["amber"], dash="dash"))
        fig.update_layout(xaxis_title="|Deviation now| (%)", yaxis_title="Off-spec probability (%)")
        ui.show(ui.style(fig, 300))
    a, b = st.columns(2)
    with a, st.container(border=True):
        ui.title("Stabilisation time by size of grade change")
        g = s.groupby("step_gsm").stabilisation_min.mean().reset_index()
        fig = go.Figure(go.Bar(x=g.step_gsm.astype(str) + " GSM", y=g.stabilisation_min, marker_color=C["cyan"]))
        fig.update_layout(yaxis_title="Minutes")
        ui.show(ui.style(fig, 280))
    with b, st.container(border=True):
        ui.title("Worst grade changes", "Top 8 by peak deviation")
        top = s.sort_values("peak_dev", ascending=False).head(8)[
            ["transition_id", "grade_from", "grade_to", "peak_dev", "off_spec_min", "stabilisation_min"]]
        ui.show_df(top.round(2))
    st.download_button("Download transition report (CSV)", s.round(3).to_csv(index=False), "gradesense_transition_report.csv")


def prediction(c: Context, S):
    page_header(c, "AI prediction", f"XGBoost forecast of basis weight deviation {HORIZON} minutes ahead")
    m = c.bundle.metrics
    ui.kpi_row([
        ("Forecast error (MAE)", f"{m['mae']:.2f}%", f"Naive 'no change' baseline {m['baseline_mae']:.2f}%", "ok"),
        ("RMSE", f"{m['rmse']:.2f}%", "On unseen grade changes", "neutral"),
        ("R²", f"{m['r2']:.2f}", f"{m['n_test']:,} held-out minutes", "info"),
        ("Off-spec recall", f"{m['recall']:.0%}", "Off-spec minutes caught early", "ok"),
        ("Off-spec precision", f"{m['precision']:.0%}", "Warnings that were real", "info"),
    ])
    st.write("")
    a, b = st.columns([3, 2])
    with a, st.container(border=True):
        ui.title("Forecast vs actual", "Selected live transition")
        ui.show(timeline_fig(c, 320))
    with b, st.container(border=True):
        ui.title("Current forecast")
        ui.show(gauge(c.p_off * 100, "Off-spec probability", ui.TONE[RISK_TONE[c.risk]]))
        ui.kv([("Forecast deviation", f"{c.pred:+.2f}%", "bad" if abs(c.pred) > c.limit else "ok"),
               ("Expected range (±1σ)", f"{c.pred - c.bundle.sigma:+.2f}% to {c.pred + c.bundle.sigma:+.2f}%", "neutral"),
               ("Risk", c.risk, RISK_TONE[c.risk])])
    te = c.df[c.df.transition_id.isin(c.bundle.test_ids) & c.df.y.notna()]
    p = c.bundle.model.predict(compute_features(te))
    with st.container(border=True):
        ui.title("Predicted vs actual on held-out transitions", "Points on the diagonal are perfect forecasts")
        n = min(2500, len(te)); idx = np.random.default_rng(0).choice(len(te), n, replace=False)
        fig = go.Figure(go.Scattergl(x=te.y.values[idx], y=p[idx], mode="markers",
                                     marker=dict(color=C["blue"], size=4, opacity=.35)))
        lo, hi = float(te.y.min()), float(te.y.max())
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], line=dict(color=C["amber"], dash="dash"), name="Ideal"))
        fig.update_layout(xaxis_title="Actual deviation (%)", yaxis_title="Predicted deviation (%)", showlegend=False)
        ui.show(ui.style(fig, 340))


def explainable(c: Context, S):
    page_header(c, "Explainable AI (SHAP)", "Why the model forecasts what it forecasts")
    ui.kpi_row([("Forecast deviation", f"{c.pred:+.2f}%", "Base value + SHAP contributions", "bad" if abs(c.pred) > c.limit else "ok"),
                ("Base value", f"{c.base:+.2f}%", "Average forecast", "neutral"),
                ("Confidence", f"{c.conf:.0%}", "In/out-of-spec call", "info")])
    st.write("")
    a, b = st.columns(2)
    with a, st.container(border=True):
        ui.title("Global importance", "Mean |SHAP| over held-out transitions")
        te = c.df[c.df.transition_id.isin(c.bundle.test_ids)].sample(1200, random_state=0)
        sv, _ = shap_values(c.bundle, compute_features(te))
        g = group_shap(sv).abs().mean().sort_values()
        fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h", marker_color=C["blue"]))
        fig.update_layout(xaxis_title="Mean |impact| on forecast (%)")
        ui.show(ui.style(fig, 340))
    with b, st.container(border=True):
        ui.title("This prediction", "How each variable moves the forecast from the base value")
        g = c.shap_g[c.shap_g.abs() > 0.005].sort_values(key=abs, ascending=False)
        fig = go.Figure(go.Waterfall(
            orientation="h", measure=["relative"] * len(g) + ["total"], y=list(g.index) + ["Forecast"],
            x=list(g.values) + [0], base=c.base, increasing=dict(marker_color=C["red"]),
            decreasing=dict(marker_color=C["blue"]), totals=dict(marker_color=C["green"])))
        fig.update_layout(xaxis_title="Deviation (%)")
        fig.update_yaxes(autorange="reversed")
        ui.show(ui.style(fig, 340))
    with st.container(border=True):
        ui.title("Key insights")
        s = c.shap_g.drop("Grade schedule").sort_values(key=abs, ascending=False)
        for k, v in s.head(4).items():
            st.markdown(f"- **{k}** {'pushes the forecast up' if v > 0 else 'pulls the forecast down'} by {abs(v):.2f}%")
        st.caption("Red = pushes deviation up (heavier sheet). Blue = pushes it down (lighter sheet).")


def recommendations(c: Context, S):
    page_header(c, "Smart recommendations", "Setpoint moves that keep the transition inside the quality threshold")
    ui.kpi_row([("Off-spec risk now", f"{c.p_off:.0%}", c.risk, RISK_TONE[c.risk]),
                ("Risk after moves", f"{c.best.p_off:.0%}", "If recommendation is applied", "ok"),
                ("Forecast now", f"{c.pred:+.2f}%", "", "neutral"),
                ("Forecast after moves", f"{c.best.pred:+.2f}%", "", "ok")])
    st.write("")
    a, b = st.columns([3, 2])
    eff = single_effects(c.bundle, c.row, c.best)
    with a, st.container(border=True):
        ui.title("Recommended setpoints")
        rows = []
        for key, k in [("steam", "steam_pct"), ("speed", "speed_pct"), ("stock", "stock_pct")]:
            cur, pct = float(c.row[key]), float(c.best[k])
            rows.append({"Variable": UNITS[key][0], "Unit": UNITS[key][1], "Current": round(cur, 2),
                         "Recommended": round(cur * (1 + pct / 100), 2), "Change": _fmt_pct(pct),
                         "Impact on deviation": f"{eff[k]:+.2f} pp"})
        ui.show_df(pd.DataFrame(rows))
        st.caption("pp = percentage points of forecast deviation, each move evaluated on its own.")
        bc, bd, be, _ = st.columns([1, 1, 1, 2])
        if bc.button("Accept", type="primary", key="fb_acc"):
            log_feedback(c, "Accepted", c.best.steam_pct, c.best.speed_pct, c.best.stock_pct)
        if bd.button("Reject", key="fb_rej"):
            log_feedback(c, "Rejected", 0, 0, 0)
        mod = be.toggle("Modify", key="fb_mod")
        if mod:
            m1, m2, m3 = st.columns(3)
            ms = m1.number_input("Steam %", -10.0, 10.0, float(c.best.steam_pct), 1.0, key="m_steam")
            mv = m2.number_input("Speed %", -10.0, 10.0, float(c.best.speed_pct), 1.0, key="m_speed")
            mk = m3.number_input("Stock %", -10.0, 10.0, float(c.best.stock_pct), 1.0, key="m_stock")
            mp = float(predict_changes(c.bundle, c.row, pd.DataFrame({"steam_pct": [ms], "speed_pct": [mv], "stock_pct": [mk]}))[0])
            st.info(f"Your version: forecast {mp:+.2f}% (off-spec probability {float(offspec_prob(mp, c.bundle.sigma, c.limit)):.0%})")
            if st.button("Save modified action", key="fb_save"):
                log_feedback(c, "Modified", ms, mv, mk)
    with b, st.container(border=True):
        ui.title("Impact on off-spec risk")
        fig = go.Figure(go.Bar(x=["Current", "After moves"], y=[c.p_off * 100, c.best.p_off * 100],
                               marker_color=[C["red"], C["green"]], text=[f"{c.p_off:.0%}", f"{c.best.p_off:.0%}"], textposition="outside"))
        fig.update_layout(yaxis_title="Off-spec probability (%)", yaxis_range=[0, 105])
        ui.show(ui.style(fig, 260))
        st.markdown("**Rationale**")
        st.markdown(copilot.answer("why this recommendation", c))
    with st.container(border=True):
        ui.title("Recent operator decisions", "Feedback is used when the model is retrained (Settings)")
        fb = S["feedback"]
        if fb:
            ui.show_df(pd.DataFrame(fb[::-1]).head(10))
        else:
            st.caption("No decisions recorded yet. Accept, reject or modify a recommendation above.")


def log_feedback(c, decision, sp, sv, sk):
    entry = {"time": dt.datetime.now().strftime("%H:%M:%S"), "transition": c.tid, "minute": c.t, "decision": decision,
             "steam %": float(sp), "speed %": float(sv), "stock %": float(sk), "forecast": round(c.pred, 2),
             "actual (+5 min)": round(float(c.row.y), 2) if pd.notna(c.row.y) else None}
    st.session_state["feedback"].append(entry)
    try:
        os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
        pd.DataFrame([entry]).to_csv(FEEDBACK_FILE, mode="a", header=not os.path.exists(FEEDBACK_FILE), index=False)
    except OSError:
        pass
    st.toast(f"{decision}: decision recorded")


def whatif(c: Context, S):
    page_header(c, "What-if simulator", "Try setpoint changes on a copy before touching the machine")
    keys = ["wi_steam", "wi_speed", "wi_stock", "wi_filler"]
    for k in keys:
        S.setdefault(k, 0.0)

    def reset():
        for k in keys:
            st.session_state[k] = 0.0

    def apply_best():
        b = st.session_state["_best"]
        st.session_state.update(wi_steam=float(b["steam_pct"]), wi_speed=float(b["speed_pct"]), wi_stock=float(b["stock_pct"]), wi_filler=0.0)

    S["_best"] = c.best
    left, right = st.columns([1, 2.4])
    with left, st.container(border=True):
        ui.title("Adjust process parameters", "Change from the current setpoint (%)")
        vals = {}
        for key, label in [("steam", "Steam pressure"), ("speed", "Machine speed"), ("stock", "Stock flow"), ("filler", "Filler flow")]:
            vals[key] = st.slider(label, -10.0, 10.0, step=0.5, key=f"wi_{key}")
            st.caption(f"{c.row[key]:.2f} → {c.row[key] * (1 + vals[key] / 100):.2f} {UNITS[key][1]}")
        b1, b2 = st.columns(2)
        b1.button("Reset to current", on_click=reset)
        b2.button("Use recommended", on_click=apply_best, type="primary")
    grid = pd.DataFrame({f"{k}_pct": [v] for k, v in vals.items()})
    wp = float(predict_changes(c.bundle, c.row, grid)[0])
    wprob = float(offspec_prob(wp, c.bundle.sigma, c.limit))
    west = est_steady_minutes(c.trans, c.t, wp)
    with right:
        ui.kpi_row([("Predicted deviation", f"{wp:+.2f}%", f"Now {c.pred:+.2f}%", "ok" if abs(wp) <= c.limit else "bad"),
                    ("Off-spec risk", f"{wprob:.0%}", f"Now {c.p_off:.0%}", RISK_TONE[risk_label(wprob)]),
                    ("Stabilisation (est.)", f"{west:.1f} min", f"Now {c.est:.1f} min", "info"),
                    ("Confidence", f"{max(wprob, 1 - wprob):.0%}", "In/out-of-spec call", "info")])
        with st.container(border=True):
            ui.title("Projected path", f"Interpolated from today's deviation to the +{HORIZON} min forecast")
            xs = np.linspace(0, HORIZON, 40)
            curve = lambda end: c.row.dev_pct + (end - c.row.dev_pct) * (1 - np.exp(-xs / 2.0)) / (1 - np.exp(-HORIZON / 2.0))
            fig = go.Figure()
            fig.add_hrect(y0=-c.limit, y1=c.limit, fillcolor=C["green"], opacity=.06, line_width=0)
            fig.add_trace(go.Scatter(x=xs, y=curve(c.pred), name="Current settings", line=dict(color=C["muted"], dash="dot")))
            fig.add_trace(go.Scatter(x=xs, y=curve(wp), name="What-if", line=dict(color=C["blue"], width=3)))
            ui.limit_lines(fig, c.limit)
            fig.update_layout(xaxis_title="Minutes from now", yaxis_title="Deviation (%)")
            ui.show(ui.style(fig, 280))
        sc = st.columns([1, 1, 3])
        if sc[0].button("Save scenario"):
            S["scenarios"].append({"name": f"Scenario {len(S['scenarios']) + 1}", **{f"{k} %": v for k, v in vals.items()},
                                   "forecast %": round(wp, 2), "off-spec risk": f"{wprob:.0%}", "stabilisation min": round(west, 1)})
        if sc[1].button("Clear saved"):
            S["scenarios"].clear()
        if S["scenarios"]:
            ui.show_df(pd.DataFrame(S["scenarios"]))
    st.caption("Estimates come from the XGBoost model; moisture is adjusted with the steam change. Treat results as decision support, not a control command.")


def copilot_page(c: Context, S):
    page_header(c, "AI manufacturing copilot", "Ask about the forecast, the risk and what to change")
    left, right = st.columns([2.2, 1])
    with right, st.container(border=True):
        ui.title("Quick questions")
        for q in copilot.SUGGESTIONS:
            if st.button(q, key=f"q_{q}"):
                S["chat"] += [("user", q), ("assistant", copilot.answer(q, c))]
        st.write("")
        ui.title("Snapshot")
        ui.kv([("Deviation", f"{c.row.dev_pct:+.2f}%", "bad" if abs(c.row.dev_pct) > c.limit else "ok"),
               ("Forecast", f"{c.pred:+.2f}%", "bad" if abs(c.pred) > c.limit else "ok"),
               ("Risk", c.risk, RISK_TONE[c.risk]), ("Confidence", f"{c.conf:.0%}", "info")])
    with left:
        if not S["chat"]:
            S["chat"].append(("assistant", "Hello, I'm your manufacturing copilot. I can explain the current forecast, "
                                           "the off-spec risk and the recommended setpoint moves."))
        for role, msg in S["chat"]:
            with st.chat_message(role):
                st.write(msg)
        if prompt := st.chat_input("Ask about the grade transition"):
            S["chat"] += [("user", prompt), ("assistant", copilot.answer(prompt, c))]
            st.rerun()


def alerts(c: Context, S):
    page_header(c, "Alerts", "Early-warning events for the selected grade change")
    tr = c.trans.iloc[:c.t + 1].copy()
    tr["pred"] = c.trans_pred[:c.t + 1]
    tr["p"] = offspec_prob(tr.pred.values, c.bundle.sigma, c.limit)
    tr["level"] = [risk_label(x) for x in tr.p]
    events, cur = [], None
    for _, r in tr.iterrows():
        if r.level != "LOW":
            if cur is None:
                cur = {"start": int(r.t), "peak": r.pred, "level": r.level}
            cur["end"] = int(r.t)
            cur["level"] = "HIGH" if "HIGH" in (cur["level"], r.level) else "MEDIUM"
            cur["peak"] = r.pred if abs(r.pred) > abs(cur["peak"]) else cur["peak"]
        elif cur:
            events.append(cur); cur = None
    if cur:
        events.append(cur)
    ui.kpi_row([("Active risk", c.risk, f"Minute {c.t}", RISK_TONE[c.risk]),
                ("Alerts this transition", str(len(events)), f"{sum(e['level'] == 'HIGH' for e in events)} high severity", "warn"),
                ("Operator decisions", str(len(S["feedback"])), "Accepted, rejected or modified", "info")])
    st.write("")
    with st.container(border=True):
        ui.title("Alert log")
        if events:
            ui.show_df(pd.DataFrame([{"Severity": e["level"], "From minute": e["start"], "To minute": e["end"],
                                      "Peak forecast": f"{e['peak']:+.2f}%",
                                      "Message": f"Forecast deviation reached {e['peak']:+.2f}% (limit ±{c.limit:g}%)"} for e in events[::-1]]))
        else:
            st.caption("No alerts so far. The forecast has stayed within the quality threshold.")
    with st.container(border=True):
        ui.title("Historical high-risk grade changes", "Largest peak deviations")
        ui.show_df(c.summary.sort_values("peak_dev", ascending=False).head(8).round(2))


def data_explorer(c: Context, S):
    ui.header("Data explorer", "Inspect, validate and replace the process data", [("Data validation active", "ok")])
    r = S["report"]
    ui.kpi_row([("Rows", f"{r['rows']:,}", f"{r['transitions']} transitions", "neutral"),
                ("Gaps interpolated", f"{r['missing_filled']}", "Sensor dropouts", "info"),
                ("Spikes repaired", f"{r['spikes_repaired']}", "Median-filter outliers", "warn")])
    st.write("")
    with st.container(border=True):
        ui.title("Process variables for a transition")
        tid = st.selectbox("Transition", sorted(c.df.transition_id.unique()), index=sorted(c.df.transition_id.unique()).index(c.tid))
        g = c.df[c.df.transition_id == tid]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.08, row_heights=[.6, .4])
        fig.add_trace(go.Scatter(x=g.t, y=g.basis_weight, name="Basis weight (GSM)", line=dict(color=C["green"])), 1, 1)
        fig.add_trace(go.Scatter(x=g.t, y=g.target, name="Target trajectory", line=dict(color=C["amber"], dash="dash")), 1, 1)
        for col, color in [("stock", C["blue"]), ("speed", C["cyan"]), ("steam", C["red"])]:
            base = g[col].iloc[0]
            fig.add_trace(go.Scatter(x=g.t, y=(g[col] / base - 1) * 100, name=f"{UNITS[col][0]} (% vs start)", line=dict(color=color)), 2, 1)
        ui.show(ui.style(fig, 420))
    with st.container(border=True):
        ui.title("Raw table")
        ui.show_df(g.drop(columns=["y"]).round(3))
        st.download_button("Download dataset (CSV)", c.df.drop(columns=["pred"], errors="ignore").to_csv(index=False), "gradesense_data.csv")
    with st.container(border=True):
        ui.title("Use your own QCS / DCS data", "CSV with these columns; it is validated, cleaned and the model is retrained")
        st.code(", ".join(REQUIRED_COLS), language="text")
        up = st.file_uploader("Upload CSV", type="csv")
        if up is not None and st.button("Validate, clean and retrain"):
            try:
                with st.spinner("Cleaning data and retraining..."):
                    df, rep = prepare(pd.read_csv(up))
                    install(S, df, rep, train_model(df, limit=S["limit"]))
                st.success("Dataset replaced. Open the Dashboard.")
            except Exception as e:  # noqa: BLE001 - surface any schema problem to the user
                st.error(f"Could not use this file: {e}")


def settings(c: Context, S):
    ui.header("Settings", "Quality threshold, recommendation limits and model lifecycle")
    a, b = st.columns(2)
    with a, st.container(border=True):
        ui.title("Quality and recommendation limits")
        S["limit"] = st.slider("Quality threshold (± % basis weight deviation)", 1.0, 5.0, S["limit"], 0.25)
        S["max_steam"] = st.slider("Max steam pressure move (%)", 2, 10, int(S["max_steam"]), 2)
        S["max_speed"] = st.slider("Max machine speed move (%)", 2, 10, int(S["max_speed"]), 2)
        S["max_stock"] = st.slider("Max stock flow move (%)", 1, 8, int(S["max_stock"]), 1)
    with b, st.container(border=True):
        ui.title("Continuous learning", "Retrain on new production data and operator feedback")
        m = c.bundle.metrics
        st.markdown(f"Current model: MAE **{m['mae']:.3f}%**, R² **{m['r2']:.2f}**, trained on {len(S['df'][S['df'].y.notna()]):,} minutes.")
        st.markdown(f"Operator decisions logged: **{len(S['feedback'])}**")
        n_new = st.slider("New grade changes to add (simulated production data)", 20, 150, 60, 10)
        if st.button("Retrain model", type="primary"):
            with st.spinner("Retraining..."):
                new, rep = prepare(generate_dataset(n_new, seed=int(dt.datetime.now().timestamp()) % 10_000, id_offset=int(S["df"].transition_id.max()) + 1))
                merged = pd.concat([S["df"].drop(columns=["pred"], errors="ignore"), new], ignore_index=True)
                old = m
                nb = train_model(merged, test_ids=S["bundle"].test_ids, limit=S["limit"])
                install(S, merged, S["report"], nb, keep_live=True)
            st.success(f"Retrained. MAE {old['mae']:.3f}% → {nb.metrics['mae']:.3f}% on the same held-out transitions.")
    st.caption("Feedback entries are also written to data/feedback_log.csv so they can be used offline.")


def install(S, df, report, bundle, keep_live=False):
    """Store dataset + model in session and refresh derived tables."""
    S["df"], S["report"], S["bundle"] = df, report, bundle
    S["summary"] = transition_summary(df, S["limit"])
    if not keep_live:
        S["tid"], S["minute"] = pick_live(df, bundle)


def pick_live(df, bundle):
    """Default scenario: held-out transition with the clearest early-warning moment."""
    best = (None, 0, -1)
    for tid in sorted(bundle.test_ids):
        tr = df[df.transition_id == tid].reset_index(drop=True)
        p = np.abs(bundle.model.predict(compute_features(tr)))
        i = int(np.argmax(p[:35]))
        if p[i] > best[2]:
            best = (tid, i, p[i])
    return int(best[0]), int(best[1])


PAGES = {"Dashboard": dashboard, "Analytics": analytics, "Prediction": prediction, "Explainable AI": explainable,
         "Recommendations": recommendations, "What-if simulator": whatif, "AI copilot": copilot_page,
         "Alerts": alerts, "Data explorer": data_explorer, "Settings": settings}

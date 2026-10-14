"""Data layer: synthetic QCS/DCS-style grade-change data, validation and cleaning.

The generator is a small physics-inspired simulator of a paper machine during a
grade change. Replace it with real QCS / DCS / Historian exports by uploading a
CSV with the columns listed in REQUIRED_COLS (Data Explorer page).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GRADES = [60, 70, 80, 90, 100, 110, 120]  # basis weight grades (GSM)
HORIZON = 5          # forecast horizon in minutes
STEPS = 60           # minutes simulated per transition
TAU = 4.0            # sheet-forming lag (minutes)
WIDTH_M = 8.0        # machine width used for the paper-waste estimate
MOIST_PER_BAR = 1.2  # moisture points removed per bar of extra steam pressure
SENSOR_COLS = ["speed", "steam", "stock", "filler", "moisture", "ash", "caliper"]

REQUIRED_COLS = [
    "transition_id", "t", "grade_from", "grade_to", "plan", "plan_next5", "target",
    "speed", "steam", "stock", "filler", "moisture", "ash", "caliper",
    "basis_weight", "dev_pct",
]


def nominal(plan):
    """Nominal setpoints for a planned basis weight (GSM)."""
    plan = np.asarray(plan, dtype=float)
    speed = 1100.0 - 3.0 * (plan - 80.0)           # m/min
    stock = plan * 1.5 * (speed / 1050.0)          # m3/h
    steam = 6.0 + 0.012 * (plan - 80.0)            # bar
    filler = plan * 1.5                            # L/min
    return {"speed": speed, "stock": stock, "steam": steam, "filler": filler}


def _ramp(t, start, dur):
    return np.clip((t - start) / max(dur, 3.0), 0.0, 1.0)


def _ar1(rng, n, phi, sigma):
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, sigma)
    return x


def simulate_transition(rng, tid, steps=STEPS):
    i0 = int(rng.integers(len(GRADES)))
    i1 = int(np.clip(i0 + rng.choice([-2, -1, 1, 2], p=[.15, .35, .35, .15]), 0, len(GRADES) - 1))
    if i1 == i0:
        i1 = i0 + 1 if i0 == 0 else i0 - 1
    g0, g1 = GRADES[i0], GRADES[i1]

    t = np.arange(steps, dtype=float)
    D = rng.uniform(14, 26)                              # planned ramp length
    plan = g0 + (g1 - g0) * _ramp(t, 2, D)
    n0, n1, npl = nominal(g0), nominal(g1), nominal(plan)

    # Operator execution errors, partly trimmed out later in the transition.
    err = {"stock": rng.normal(0, .017), "speed": rng.normal(0, .012),
           "steam": rng.normal(0, .07), "filler": rng.normal(0, .03)}
    trimmed = rng.random() < 0.6
    trim_f = 1 - 0.85 * trimmed * _ramp(t, 2 + D + rng.uniform(6, 14), 5)
    sp = {}
    for k in ["stock", "speed", "steam", "filler"]:
        q = _ramp(t, 2 + rng.normal(0, 1.5), D * rng.uniform(.85, 1.15))
        sp[k] = n0[k] + (n1[k] * (1 + err[k] * trim_f) - n0[k]) * q

    moist = 6.8 - MOIST_PER_BAR * (sp["steam"] - npl["steam"]) + _ar1(rng, steps, .9, .05)
    dist = _ar1(rng, steps, .9, .0025)                   # unmeasured furnish disturbance
    eq = ((sp["stock"] / 1.5) * (1050.0 / sp["speed"])
          * (1 - 0.015 * (sp["steam"] - npl["steam"]))
          * (1 + 0.006 * (moist - 6.8))
          * (1 + 0.15 * (sp["filler"] / npl["filler"] - 1))
          * (1 + dist))

    bw = np.empty(steps); tg = np.empty(steps)
    bw[0] = tg[0] = g0
    for i in range(1, steps):
        bw[i] = bw[i - 1] + (eq[i] - bw[i - 1]) / TAU
        tg[i] = tg[i - 1] + (plan[i] - tg[i - 1]) / TAU
    bw_meas = bw * (1 + rng.normal(0, .0025, steps))

    df = pd.DataFrame({
        "transition_id": tid, "t": t.astype(int), "grade_from": g0, "grade_to": g1,
        "plan": plan, "target": tg,
        "speed": sp["speed"] * (1 + rng.normal(0, .002, steps)),
        "steam": sp["steam"] * (1 + rng.normal(0, .004, steps)),
        "stock": sp["stock"] * (1 + rng.normal(0, .002, steps)),
        "filler": sp["filler"] * (1 + rng.normal(0, .004, steps)),
        "moisture": moist + rng.normal(0, .08, steps),
        "ash": 14.0 * (sp["filler"] / npl["filler"]) ** 0.8 + rng.normal(0, .15, steps),
        "caliper": 1.25 * tg * (1 + rng.normal(0, .01, steps)),
        "basis_weight": bw_meas,
        "dev_pct": (bw_meas - tg) / tg * 100,
    })
    df["plan_next5"] = df["plan"].shift(-HORIZON).fillna(g1)
    return df


def generate_dataset(n_transitions=248, seed=7, id_offset=0, inject_faults=True):
    """Simulate n grade changes (default 248, as in the project deck)."""
    rng = np.random.default_rng(seed)
    df = pd.concat([simulate_transition(rng, id_offset + i) for i in range(n_transitions)],
                   ignore_index=True)
    if inject_faults:  # sensor dropouts and spikes so the validation pipeline has work to do
        for c in SENSOR_COLS:
            m = rng.random(len(df))
            df.loc[m < .004, c] = np.nan
            spikes = (m >= .004) & (m < .0055)
            df.loc[spikes, c] = df.loc[spikes, c] * rng.choice([.5, 1.6], spikes.sum())
    return df


def validate_and_clean(df):
    """Data validation pipeline: schema check, interpolate gaps, repair spikes."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    df = df.sort_values(["transition_id", "t"]).reset_index(drop=True).copy()
    n_missing = int(df[SENSOR_COLS].isna().sum().sum())
    n_spikes = 0
    for c in SENSOR_COLS:
        g = df.groupby("transition_id")[c]
        df[c] = g.transform(lambda s: s.interpolate(limit_direction="both"))
        med = g.transform(lambda s: s.rolling(5, center=True, min_periods=1).median())
        bad = (df[c] - med).abs() / med.abs().clip(lower=1e-6) > 0.20
        n_spikes += int(bad.sum())
        df.loc[bad, c] = med[bad]
    df = df.dropna(subset=["basis_weight", "dev_pct"]).reset_index(drop=True)
    return df, {"rows": len(df), "transitions": int(df.transition_id.nunique()),
                "missing_filled": n_missing, "spikes_repaired": n_spikes}


def add_deltas_and_labels(df):
    """Recent setpoint movement (last 5 min) and the 5-minute-ahead target label."""
    df = df.sort_values(["transition_id", "t"]).reset_index(drop=True)
    g = df.groupby("transition_id")
    for src, dst in [("stock", "d_stock"), ("speed", "d_speed"), ("steam", "d_steam")]:
        df[dst] = ((df[src] - g[src].shift(HORIZON)) / df[src] * 100).fillna(0.0)
    df["y"] = g["dev_pct"].shift(-HORIZON)
    return df


def prepare(df):
    df, report = validate_and_clean(df)
    return add_deltas_and_labels(df), report


def transition_summary(df, limit=2.5):
    """One row per grade change: peak deviation, off-spec minutes, stabilisation time, waste."""
    rows = []
    ton_per_min = df["speed"] * WIDTH_M * df["basis_weight"] / 1e6
    for tid, g in df.groupby("transition_id"):
        out = g["dev_pct"].abs() > limit
        last_out = g.loc[out, "t"].max() if out.any() else -1
        rows.append({
            "transition_id": tid, "grade_from": g.grade_from.iloc[0], "grade_to": g.grade_to.iloc[0],
            "step_gsm": abs(g.grade_to.iloc[0] - g.grade_from.iloc[0]),
            "peak_dev": g["dev_pct"].abs().max(), "off_spec_min": int(out.sum()),
            "stabilisation_min": int(last_out + 1),
            "waste_tons": float(ton_per_min.loc[g.index][out].sum()),
            "off_spec": bool(out.sum() > 3),
        })
    return pd.DataFrame(rows)

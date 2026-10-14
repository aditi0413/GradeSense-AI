"""Prediction, risk, SHAP explanations, recommendation engine and what-if logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, r2_score

from .data import HORIZON, MOIST_PER_BAR, nominal

FEATURES = [
    "dev_now", "target", "plan_change_5", "stock_rel", "speed_rel", "steam_rel",
    "filler_rel", "moisture", "ash", "caliper_rel", "d_stock", "d_speed", "d_steam", "t",
]
# Physical direction of influence on basis-weight deviation (keeps what-if answers sane).
MONOTONE = {"dev_now": 1, "stock_rel": 1, "speed_rel": -1, "steam_rel": -1,
            "filler_rel": 1, "moisture": 1}
GROUPS = {
    "Current deviation": ["dev_now"],
    "Stock flow": ["stock_rel", "d_stock"],
    "Machine speed": ["speed_rel", "d_speed"],
    "Steam pressure": ["steam_rel", "d_steam"],
    "Moisture content": ["moisture"],
    "Filler flow": ["filler_rel"],
    "Ash content": ["ash"],
    "Caliper": ["caliper_rel"],
    "Grade schedule": ["plan_change_5", "target", "t"],
}


def compute_features(raw: pd.DataFrame) -> pd.DataFrame:
    nom = nominal(raw["plan"].values)
    f = pd.DataFrame(index=raw.index)
    f["dev_now"] = raw["dev_pct"]
    f["target"] = raw["target"]
    f["plan_change_5"] = (raw["plan_next5"] / raw["plan"] - 1) * 100
    f["stock_rel"] = (raw["stock"] / nom["stock"] - 1) * 100
    f["speed_rel"] = (raw["speed"] / nom["speed"] - 1) * 100
    f["steam_rel"] = (raw["steam"] / nom["steam"] - 1) * 100
    f["filler_rel"] = (raw["filler"] / nom["filler"] - 1) * 100
    f["moisture"] = raw["moisture"]
    f["ash"] = raw["ash"]
    f["caliper_rel"] = (raw["caliper"] / (1.25 * raw["target"]) - 1) * 100
    for c in ["d_stock", "d_speed", "d_steam"]:
        f[c] = raw[c]
    f["t"] = raw["t"]
    return f[FEATURES]


@dataclass
class Bundle:
    model: xgb.XGBRegressor
    sigma: float
    metrics: dict
    test_ids: set
    explainer: object = field(default=None, repr=False)


def split_ids(df, seed=42, frac=0.2):
    ids = np.array(sorted(df.transition_id.unique()))
    np.random.default_rng(seed).shuffle(ids)
    n = int(len(ids) * frac)
    return set(ids[n:].tolist()), set(ids[:n].tolist())


def evaluate(model, te, limit=2.5):
    pred = model.predict(compute_features(te))
    y = te["y"].values
    act, prd = np.abs(y) > limit, np.abs(pred) > limit
    tp = int((act & prd).sum())
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "r2": float(r2_score(y, pred)),
        "baseline_mae": float(mean_absolute_error(y, te["dev_pct"].values)),
        "recall": tp / max(int(act.sum()), 1),
        "precision": tp / max(int(prd.sum()), 1),
        "n_test": len(te),
    }, pred


def train_model(df, test_ids=None, seed=42, limit=2.5) -> Bundle:
    if test_ids is None:
        _, test_ids = split_ids(df, seed)
    lab = df[df["y"].notna()]
    tr, te = lab[~lab.transition_id.isin(test_ids)], lab[lab.transition_id.isin(test_ids)]
    mono = "(" + ",".join(str(MONOTONE.get(f, 0)) for f in FEATURES) + ")"
    model = xgb.XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.85,
        colsample_bytree=0.9, min_child_weight=5, reg_lambda=2.0,
        monotone_constraints=mono, random_state=seed, n_jobs=2)
    model.fit(compute_features(tr), tr["y"])
    metrics, pred = evaluate(model, te, limit)
    return Bundle(model, float(np.std(te["y"].values - pred)), metrics, set(test_ids))


def offspec_prob(pred, sigma, limit=2.5):
    """P(|deviation| > limit) assuming forecast error ~ N(0, sigma)."""
    pred = np.asarray(pred, dtype=float)
    return 1 - (norm.cdf((limit - pred) / sigma) - norm.cdf((-limit - pred) / sigma))


def risk_label(p):
    return "LOW" if p < 0.2 else "MEDIUM" if p < 0.5 else "HIGH"


def shap_values(bundle: Bundle, X: pd.DataFrame):
    """SHAP values (TreeExplainer; falls back to XGBoost's exact native contributions)."""
    try:
        import shap
        if bundle.explainer is None:
            bundle.explainer = shap.TreeExplainer(bundle.model)
        sv = bundle.explainer.shap_values(X)
        base = float(np.ravel(bundle.explainer.expected_value)[0])
    except Exception:
        c = bundle.model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        sv, base = c[:, :-1], float(c[0, -1])
    return pd.DataFrame(sv, columns=FEATURES, index=X.index), base


def group_shap(sv: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({g: sv[cols].sum(axis=1) for g, cols in GROUPS.items()})


# ---- what-if & recommendations -------------------------------------------------
def apply_changes(row: pd.Series, grid: pd.DataFrame) -> pd.DataFrame:
    """Apply %-changes to controllable setpoints (steam, speed, stock, filler)."""
    raw = pd.DataFrame([row.to_dict()] * len(grid))
    for col, key in [("steam", "steam_pct"), ("speed", "speed_pct"),
                     ("stock", "stock_pct"), ("filler", "filler_pct")]:
        if key in grid:
            raw[col] = row[col] * (1 + grid[key].values / 100)
    raw["moisture"] = row["moisture"] - MOIST_PER_BAR * (raw["steam"] - row["steam"])
    return raw


def predict_changes(bundle, row, grid):
    return bundle.model.predict(compute_features(apply_changes(row, grid)))


def recommend(bundle, row, limit=2.5, max_steam=6, max_speed=6, max_stock=4):
    """Grid-search setpoint moves that minimise the forecast deviation with a small
    penalty on move size (operators prefer small changes)."""
    ax = lambda m, s: np.arange(-m, m + 1e-9, s)
    combos = list(product(ax(max_steam, 2), ax(max_speed, 2), ax(max_stock, 1)))
    grid = pd.DataFrame(combos, columns=["steam_pct", "speed_pct", "stock_pct"])
    grid["pred"] = predict_changes(bundle, row, grid)
    grid["p_off"] = offspec_prob(grid["pred"], bundle.sigma, limit)
    grid["effort"] = grid.steam_pct.abs() / 2 + grid.speed_pct.abs() / 2 + grid.stock_pct.abs()
    grid["score"] = grid["pred"].abs() + 0.03 * (grid.steam_pct.abs() + grid.speed_pct.abs() + grid.stock_pct.abs())
    return grid.sort_values("score").reset_index(drop=True)


def single_effects(bundle, row, best):
    """Impact of each recommended move on its own (percentage points of deviation)."""
    base = float(predict_changes(bundle, row, pd.DataFrame({"steam_pct": [0.0]}))[0])
    out = {}
    for k in ["steam_pct", "speed_pct", "stock_pct"]:
        g = pd.DataFrame({k: [float(best[k])]})
        out[k] = float(predict_changes(bundle, row, g)[0]) - base
    return out


def est_steady_minutes(trans: pd.DataFrame, t: int, pred: float):
    """Heuristic: remaining planned ramp + ~3 min of settling per % above 1% deviation."""
    g1 = trans["grade_to"].iloc[0]
    done = trans.loc[(trans["plan"] - g1).abs() < 0.01, "t"]
    remaining = max(0, (int(done.min()) if len(done) else len(trans)) - t)
    return remaining + 3.0 * max(0.0, abs(pred) - 1.0)

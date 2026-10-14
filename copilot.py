"""Rule-based operator copilot. Every answer is assembled from the live SHAP values,
forecast and recommendation, so nothing is invented. Swap `answer` for an LLM call
if you want free-form chat."""
from __future__ import annotations

SUGGESTIONS = ["Why is the deviation high?", "What adjustments do you recommend?",
               "Will this go off-spec?", "Why this recommendation?"]


def _actions(c):
    names = {"steam_pct": "steam pressure", "speed_pct": "machine speed", "stock_pct": "stock flow"}
    out = []
    for k, n in names.items():
        v = float(c.best[k])
        if abs(v) > 1e-9:
            out.append(f"{'Reduce' if v < 0 else 'Increase'} {n} by {abs(v):.0f}%")
    return out


def _top(c, n=3):
    s = c.shap_g.drop("Grade schedule", errors="ignore")
    s = s.reindex(s.abs().sort_values(ascending=False).index)[:n]
    return [f"{k.lower()} ({'raises' if v > 0 else 'lowers'} the forecast by {abs(v):.2f}%)" for k, v in s.items()]


def answer(q: str, c) -> str:
    q = q.lower()
    status = "outside" if abs(c.row.dev_pct) > c.limit else "within"
    if any(w in q for w in ["why this", "justify", "rationale", "reason"]):
        acts = _actions(c)
        if not acts:
            return "No move is recommended: the forecast is already close to target, so changes would add risk without benefit."
        return (f"I searched {len(c.rec)} combinations of steam, speed and stock-flow moves, scored by forecast deviation "
                f"plus a small penalty for larger moves. The best was: {'; '.join(acts)}. "
                f"It lowers the 5-minute forecast from {c.pred:+.2f}% to {c.best.pred:+.2f}% "
                f"(off-spec probability {c.p_off:.0%} → {c.best.p_off:.0%}).")
    if any(w in q for w in ["adjust", "recommend", "what should", "what do i do", "action"]):
        acts = _actions(c)
        if not acts:
            return "Hold the current setpoints. The forecast stays inside the quality threshold."
        return (f"I recommend: {'; '.join(acts)}. Expected forecast {c.pred:+.2f}% → {c.best.pred:+.2f}%, "
                f"off-spec risk {c.p_off:.0%} → {c.best.p_off:.0%}. You can accept, reject or modify this on the Recommendations page.")
    if any(w in q for w in ["off-spec", "offspec", "risk", "will", "threshold", "go out"]):
        return (f"The 5-minute forecast is {c.pred:+.2f}% against a ±{c.limit:g}% limit. "
                f"Probability of leaving spec is {c.p_off:.0%} ({c.risk.lower()} risk). Current deviation is {c.row.dev_pct:+.2f}% ({status} spec).")
    if any(w in q for w in ["why", "deviation", "contribut", "driver", "cause", "explain"]):
        return (f"Current deviation is {c.row.dev_pct:+.2f}% ({status} the ±{c.limit:g}% limit) and the forecast is {c.pred:+.2f}%. "
                f"The main drivers, from SHAP, are: " + "; ".join(_top(c)) + ".")
    if "what-if" in q or "simulat" in q:
        return "Open the What-if Simulator, move the steam, speed or stock-flow sliders, and compare the predicted outcome with the current one before touching the machine."
    return ("I can explain the deviation, the off-spec risk and the recommended adjustments. "
            "Try: " + " / ".join(f"“{s}”" for s in SUGGESTIONS))

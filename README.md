# GradeSense AI

AI-powered grade change intelligence for paper manufacturing: predicts basis weight deviation
5 minutes ahead, warns before the ±2.5% quality threshold is crossed, recommends setpoints,
explains every prediction (SHAP) and lets operators simulate changes first.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

First launch simulates 248 grade changes and trains the model (a few seconds).

## What maps to the deck

| Deck item | Where |
|---|---|
| Real-time grade monitoring dashboard | Dashboard |
| Production analytics, basis weight distribution and risk correlation | Analytics |
| XGBoost prediction engine, off-spec and risk level | Prediction |
| SHAP explainability, correlation discovery | Explainable AI |
| Recommendation engine, accept / reject / modify feedback loop | Recommendations |
| What-if simulator | What-if simulator |
| AI manufacturing copilot | AI copilot |
| Alerts and notifications | Alerts |
| Data validation pipeline, bring-your-own QCS data | Data explorer |
| Continuous learning (retrain on new data + feedback) | Settings |

## Layout

```
app.py                 entry point, navigation, live-scenario controls
gradesense/data.py     simulator, validation/cleaning, transition summary
gradesense/model.py    features, XGBoost, risk, SHAP, recommendations, what-if
gradesense/copilot.py  rule-based copilot built from live SHAP + forecast
gradesense/views.py    all pages
gradesense/ui.py       theme, KPI cards, Plotly styling
tests/test_smoke.py    renders every page headlessly (pytest)
```

## Using real plant data

Upload a CSV in Data explorer with these columns: `transition_id, t, grade_from, grade_to, plan,
plan_next5, target, speed, steam, stock, filler, moisture, ash, caliper, basis_weight, dev_pct`
(`plan` = planned GSM trajectory, `target` = expected basis weight, `dev_pct` = (basis_weight - target) / target × 100).
It is validated, gaps are interpolated, spikes repaired, and the model retrains.

## Honest notes

- The bundled data is **simulated** (physics-inspired, with sensor dropouts and spikes). The deck's
  numbers (28% off-spec, 92% confidence, 3.00% → 1.40%) were illustrative; this app computes its own.
  On the simulated hold-out set the model's MAE is about 0.46% vs 0.51% for a "no change" baseline,
  so treat accuracy claims as a property of the simulator until you train on plant data.
- "Confidence" is the probability the in/out-of-spec call is right, from the forecast error spread.
- "Time to steady state" is a heuristic (remaining planned ramp + settling allowance).
- The copilot is rule-based on purpose (answers come from real SHAP values); swap `copilot.answer` for an LLM if wanted.
- The Honeywell Forge, PLC and live-historian items from Future Scope are not implemented.

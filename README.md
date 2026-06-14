# UFC Betting-Market Efficiency Study & Forward-Test Harness

> This repository holds my scripts and the write-up. The base open-source model's code is not included, so `tracker.py` needs that model's data pipeline to run.

A quantitative case study: I took an open-source UFC fight-prediction model, rigorously tested whether it can beat real betting markets, and built an automated forward-testing pipeline with closing-line-value (CLV) measurement. **Headline result: the UFC moneyline market is efficient relative to a public-data ML model, no exploitable edge was found, and I proved it instead of assuming it.**

> Honest framing up front: this project's value is the *methodology and the negative result*, not a profitable betting system. Most "I have a model that beats the books" projects quietly lose money; this one was stress-tested to a clear, data-backed conclusion without risking a cent.

---

## TL;DR

| Question | Answer | Evidence |
|---|---|---|
| Does the model predict winners better than chance? | Yes | 62% out-of-sample accuracy vs 53-55% base rate |
| Does it beat the **market**? | **No** | Market 70% accuracy vs model 62%; market log loss 0.58 vs 0.65 |
| Does value-betting its edges make money? | **No** | Backtest ROI **−9.5%** (95% CI [−18.9%, 0], P(profit)=3%) |
| Is the model's "edge" signal informative? | **No (anti-informative)** | Higher claimed edge → worse ROI (edge 10-15% bucket: −17%) |
| Where could edge still exist? | Round/prop markets at soft books, **unproven**, forward-test only |

---

## What I did

1. **Validated the source model.** Verified the open-source repo runs end-to-end and reproduces its predictions.
2. **Sourced real odds.** Historical moneyline odds for ~7,000 fights (2010-2026) + live odds across 19 sportsbooks via The Odds API (including Pinnacle as a sharp benchmark).
3. **Ran two independent backtests** of increasing sophistication, both out-of-sample against real prices.
4. **Engineered improvements**, Elo ratings (strength-of-schedule), isotonic probability calibration, an edge cap, and a round-distribution model for over/under markets.
5. **Built an automated forward-test harness** that logs model picks + odds, captures closing lines, computes CLV, benchmarks against the sharp book, and settles results, runs on a schedule, unattended.
6. **Documented every finding honestly**, including the ones that killed the thesis.

---

## Key findings

### 1. The market beats the model (out-of-sample, 1,645 fights, 2023-2026)

| | Accuracy | Log loss |
|---|---|---|
| Model (XGBoost + Elo + calibration) | 61.8% | 0.654 |
| **Market (de-vigged)** | **70.0%** | **0.583** |

### 2. Value-betting the model loses money

| Strategy | Bets | ROI |
|---|---|---|
| EV > 5%, edge-capped | 432 | **−9.5%** (95% CI [−18.9%, 0]) |
| EV > 10% | 324 | −11.0% |
| Bet favorite every fight | 1,645 | −0.8% |

### 3. The "edge" signal is anti-predictive

| Model-claimed edge | Realized ROI |
|---|---|
| 5-10% | −4.8% |
| 10-15% | **−17.1%** |
| 15-30% | −7.9% |

The fights where the model most disagrees with the market are where it is *most wrong*, a textbook signature of a model weaker than the price-setter. (Diagnosed concretely as **strength-of-schedule blindness**: career-aggregate features reward padded records and penalize elites who lost to other elites.)

### 4. Improvements didn't cross the bar

Elo features and isotonic calibration removed absurd predictions but did **not** close the accuracy gap to the market. Hand-engineered round-prediction features (finish propensity, pace, durability) did **not** improve out-of-sample log loss, the gradient-boosted model was already at the data ceiling (~62% on "goes the distance" vs 59.5% base).

---

## The forward-test harness (`tracker.py`)

The reusable deliverable. Each run:

- Trains a **calibrated** winner model (Elo-augmented) and a **round-distribution** model on the full history.
- Pulls **live odds** (moneyline + round totals) across 19 books via The Odds API.
- Rebuilds per-fighter features from history for any upcoming card.
- Logs only **+EV** picks (with an edge cap to filter model blow-ups) to `picks_log.csv`, tagging each with the **Pinnacle sharp probability** and `model_vs_sharp`.
- Captures **closing lines** and computes **CLV**, the one edge metric that can't be faked.
- Reconciles settled fights and prints a running ROI/CLV scorecard.
- Runs automatically on a schedule.

**Why CLV?** Beating the closing line over a large sample is the gold-standard, un-gameable proof of edge. The harness is designed so that *if* a real edge exists in round markets, CLV will reveal it, and if not, it will show that too, without betting real money.

---

## Tech stack

`Python` · `pandas` / `numpy` · `XGBoost` · `scikit-learn` (isotonic calibration, `FrozenEstimator`) · custom Elo · The Odds API · Polymarket Gamma API (evaluated) · bootstrap confidence intervals · scheduled automation.

---

## What this demonstrates

- **End-to-end ML**: data sourcing, feature engineering, calibration, time-aware validation.
- **Quant rigor**: out-of-sample backtesting, de-vigging, CLV, bootstrap CIs, multiple-comparison awareness, market-efficiency reasoning.
- **Intellectual honesty**: chasing a result to a clear negative conclusion instead of overfitting a story, and not deploying changes that failed to validate.

## Honest conclusion

A home-grown model on public data does not beat the UFC betting market. The market's prices already encode more than the model can extract from free historical stats. The remaining open question, whether softer round/prop markets are beatable, is left running as a live forward-test with CLV as the judge.

*This is a research and educational project. It is not betting advice; sports betting is negative-expectation by default.*

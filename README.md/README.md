# Trader Performance vs. Bitcoin Market Sentiment

Exploring the relationship between Bitcoin market sentiment (Fear & Greed Index) and trader performance on Hyperliquid, using EDA, statistical hypothesis testing, and predictive modeling.

## Objective

Analyze historical trader data alongside daily Bitcoin market sentiment to uncover patterns in trading behavior and performance, and evaluate whether sentiment can meaningfully predict trade outcomes.

## Datasets

| File | Description | Rows |
|---|---|---|
| `fear_greed_index.csv` | Daily Bitcoin Fear & Greed Index (2018–present): `timestamp`, `value`, `classification`, `date` | 2,645 |
| `historical_data.csv` | Trade-level Hyperliquid data (May 2023–May 2025): `Account`, `Coin`, `Execution Price`, `Size USD`, `Side`, `Direction`, `Timestamp IST`, `Closed PnL`, `Fee`, etc. | 211,224 |

Analysis window is constrained to the overlap of both datasets: **May 2023 – May 2025** (2,644 days of sentiment data, 211,224 trade-level records).

## Methodology

The analysis follows a 9-phase pipeline (see `notebook.ipynb`):

| Phase | Description |
|---|---|
| 0 | Environment setup |
| 1 | Data loading & first look |
| 2 | Cleaning & type fixes (datetime parsing, sentiment simplification) |
| 3 | Feature engineering (closing-trade flags, win flags, notional, direction) |
| 4 | Merge trade data with sentiment on date; build aggregation tables |
| 5 | Exploratory analysis (activity, win rate, PnL distribution, long/short bias, per-account & per-coin cuts) |
| 6 | Hidden-pattern analysis (lag effects, sentiment transitions, tail risk) |
| 7 | Statistical hypothesis testing (Mann-Whitney, Welch's t-test, chi-square, correlation, Kruskal-Wallis, effect sizes) |
| 8 | Predictive modeling — logistic/linear regression baselines vs. Random Forest, time-based train/test split, feature importance |
| 9 | Synthesis, findings summary, and report export |

## Key Findings

- **Win rate by sentiment:** Fear 84.40% · Neutral 82.35% · Greed 82.37%
- **Fear vs. Greed PnL difference:** Mann-Whitney p = 0.6262, rank-biserial effect size = 0.0019 → **not statistically significant**, negligible effect size
- **Long/short bias vs. sentiment:** chi-square p < 0.00001, Cramér's V = 0.1795 → statistically significant, small-to-moderate effect
- **Sentiment value vs. daily PnL correlation:** Pearson r = -0.0826 (weak, negative — not meaningfully predictive)
- **Predictive modeling:** Logistic Regression AUC = 0.7027 vs. Random Forest (balanced) AUC = 0.5292
- **Top predictive feature:** `account_rolling_winrate` (importance = 0.5834) — ranked 1st out of 6 features; sentiment-related features (`prev_sentiment_value`, importance = 0.227) ranked 2nd
- **Fee drag as % of PnL:** varies by sentiment regime, see `fee_as_pct_of_pnl` in notebook output

## Modeling Approach

- **Target:** win/loss (classification) and Closed PnL (regression), on closing trades only
- **Features:** lagged sentiment (value + class), account rolling win rate (expanding, no leakage), coin group (major vs. alt), trade side, notional size
- **Split:** time-based (train on earlier trades, test on later trades) — avoids lookahead bias
- **Models:** Logistic/Linear Regression (baseline) vs. Random Forest (class-balanced for classifier)

### Model Comparison

| Model | Task | Metric 1 | Value 1 | Metric 2 | Value 2 |
|---|---|---|---|---|---|
| Logistic Regression | win/loss | Accuracy | 0.7357 | AUC | 0.7027 |
| Random Forest Classifier | win/loss | Accuracy | 0.7387 | AUC | 0.5292 |
| Linear Regression | Closed PnL | RMSE | 1763.32 | R² | -0.2621 |
| Random Forest Regressor | Closed PnL | RMSE | 1663.98 | R² | -0.1239 |

**Caveat:** With only 32 accounts and a single macro sentiment value per day, sentiment carries limited predictive signal relative to account-specific trading history. Model performance should be read as "how much sentiment adds on top of account behavior," not as a standalone trading signal. Both PnL regression models post negative R², meaning closed PnL magnitude is not well explained by the available features — findings should be treated as directional hypotheses, not trading rules.

## Repository Structure

```
├── notebook.ipynb                  # Full analysis notebook (all 9 phases)
├── README.md                       # This file
├── merged_trades_sentiment.csv     # Cleaned, merged intermediate dataset
├── model_comparison.csv            # Baseline vs. Random Forest metrics
├── feature_importance.csv          # Feature importance from win/loss classifier
├── hypothesis_test_results.csv     # All statistical test results + effect sizes
├── tail_risk_stats.csv             # PnL distribution stats (std, skew) by sentiment
└── summary_report.md               # Auto-generated findings summary
```

## Tech Stack

Python · pandas · numpy · matplotlib · seaborn · scipy.stats · scikit-learn

## How to Run

1. Open `notebook.ipynb` in Google Colab or Jupyter.
2. Run Phase 0 to install/import dependencies.
3. Upload `fear_greed_index.csv` and `historical_data.csv` when prompted (Phase 0.3).
4. Run all cells top to bottom — later phases depend on variables created in earlier ones.

## Disclaimer

Findings are exploratory and framed as hypotheses for further testing. Nothing in this analysis constitutes financial or trading advice.

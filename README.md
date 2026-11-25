# Freight rate prediction: approach and results

## Task
Predict `posted_rate` for 12,000 validation loads (Nov-Dec 2025) from 48,000 labelled loads (Jan-Oct 2025), and
produce a 31-day December chart for one fixed lane (Lexington -> Fort Wayne, 360 mi, Dry Van, 32,000 lb).

## What the data showed
- **Rate is roughly distance x rate-per-mile** (corr 0.91), so the model predicts rate-per-mile and multiplies by distance.
- **~1.4% of labels are extreme outliers** (rate-per-mile < 1.2 or > 3.5; about 0.3x or 3.5x normal) with no pattern in
  any feature, month or equipment type. They are unpredictable, so I used an absolute-error loss (predicts the median)
  and report errors both on all rows and on non-outlier rows.
- **Data quality:** 300 missing weights (left missing; the model handles them natively), 292 negative weights (treated as
  sign errors, magnitude kept), 374 missing `market_index` (not used).
- **The validation file covers Nov AND Dec**, a ~2-month horizon, so I backtested with 2-month-ahead windows.
- **8 validation cities (about 12% of rows) never appear in training.** Geography enters as coordinates, not city names.

## Model
scikit-learn `HistGradientBoostingRegressor`, absolute-error loss, target = rate / distance.
Features: distance, |weight|, equipment, day of week, pickup/delivery latitude and longitude.

## Backtest (train before origin, test the following 61 days; MAE in $)
| Origin | Model MAE | Model MAE (non-outlier) | Baseline MAE |
|---|---|---|---|
| Jun 1 | 173.5 | 116.0 | 247.1 |
| Jul 1 | 104.8 | 52.5 | 248.8 |
| Aug 1 | 95.6 | 45.0 | 261.3 |
| Sep 1 | 98.8 | 42.8 | 257.0 |

Baseline = median rate-per-mile x distance. Typical error on non-outlier loads is about 1.8-2.2% (MAPE).
RMSE stays around $630 in every run because the random outliers dominate it; no model can predict those.

## What I tested and rejected (Sep-Oct holdout, non-outlier MAE)
| Change | Effect |
|---|---|
| Raw `market_index` per load | worse: 51 -> 59 |
| Daily/7-day smoothed `market_index` | much worse: 46 -> 110-120 (tracks seasonality in training, fails out of time) |
| `quote_signal` | worse |
| Time index / day of month | worse or neutral; unsafe to extrapolate into Nov-Dec |
| Level adjustment from recent weeks | inconsistent across origins |
| Squared-error loss | worse MAE, no RMSE gain |
| City labels instead of coordinates | ~9% worse on unseen cities |
| Day of week | kept: better or equal at all 4 origins (about +-0.9% weekly pattern) |

## Limitations
- The December chart shows only the weekly pattern (Thursday high, Sunday low, about +-1%). The model has no holiday
  or year-end information, so Dec 25 is treated like any Thursday.
- Nov-Dec level is extrapolated from the full Jan-Oct history; there is no way to validate beyond the backtest.
- The final score uses Spotter's hidden labels and metric.

## Reproduce
Place the provided files in `data/` as `train_test.csv`, `validation.csv`, `validation_predictions_template.csv` and `december_chart_inputs.csv` (the folder is git-ignored so the company's data is not published).
```
pip install -r requirements.txt
python solution.py --data-dir data --output-dir outputs
python score.py --predictions outputs/validation_predictions.csv --december-predictions outputs/december_chart_inputs_completed.csv --output-dir scorer_results
```

# Phase 9 — model report

- Data: `data\processed\laptop_prices_clean.csv` (5678 records)
- Best model: **random_forest**
- Trained at: 2026-11-10T19:44:25.242554+00:00
- scikit-learn: 1.9.1 | Python: 3.14.7
- Split sizes — train 3323 / val 942 / test 1195

MAE and RMSE are in maloti (LSL); lower is better. R² is closer to 1 for better fit.

## Model comparison (validation)
| model | n | mae | rmse | r2 |
| --- | --- | --- | --- | --- |
| median_baseline | 942 | 8609.89 | 12336.38 | -0.17 |
| linear_regression | 942 | 5728.10 | 7985.83 | 0.51 |
| random_forest | 942 | 5621.29 | 7853.84 | 0.53 |

## Model comparison (held-out test — untouched during tuning)
| model | n | mae | rmse | r2 |
| --- | --- | --- | --- | --- |
| median_baseline | 1195 | 7102.35 | 10201.55 | -0.02 |
| linear_regression | 1195 | 4369.78 | 6281.29 | 0.61 |
| random_forest | 1195 | 4241.16 | 6252.45 | 0.62 |

## Test error by price band
| band | n | mae | rmse |
| --- | --- | --- | --- |
| (13167.347, 19251.57] | 278 | 4700 | 6044 |
| (134.518, 9202.008] | 364 | 2393 | 3461 |
| (19251.57, 62961.11] | 281 | 7263 | 9718 |
| (9202.008, 13167.347] | 272 | 3123 | 4563 |

## Figures
- `reports/figures/actual_vs_predicted.png`
- `reports/figures/error_by_price_band.png`
- `reports/figures/learning_curve.png`
- `reports/figures/feature_importance.png`

## Limitations
- Metrics are only as trustworthy as the sample size; a few hundred records gives wide confidence intervals.
- Prices are **asking** prices unless the dataset was filtered to completed sales; asking and completed prices are kept separate.
- Accuracy is reported as MAE/RMSE/R², not as a single 'accuracy' percentage.

# Phase 8 — data cleaning report

- Input: `data\raw\laptop_prices_raw.csv`
- Cleaned output: `C:\Users\dalnk\Downloads\LUCT-Market-Phase-7-Sepolia-Integration\luct-ai\data\processed\laptop_prices_clean.csv`
- Records after cleaning: **5678**
- Duplicates removed: **3157**

## Notes
- input records: 8835
- duplicates removed: 3157
- usable records after cleaning: 5678

## Missing values (before cleaning)
| column | missing | present | missing_pct |
| --- | --- | --- | --- |
| record_id | 0 | 8835 | 0.00 |
| brand | 0 | 8835 | 0.00 |
| model | 8835 | 0 | 100.00 |
| processor | 0 | 8835 | 0.00 |
| ram_gb | 0 | 8835 | 0.00 |
| storage_gb | 2 | 8833 | 0.02 |
| storage_type | 0 | 8835 | 0.00 |
| age_years | 8835 | 0 | 100.00 |
| condition | 6304 | 2531 | 71.35 |
| battery_health | 8835 | 0 | 100.00 |
| price | 497 | 8338 | 5.63 |
| currency | 0 | 8835 | 0.00 |
| source | 0 | 8835 | 0.00 |
| collection_date | 0 | 8835 | 0.00 |
| price_type | 0 | 8835 | 0.00 |
| notes | 0 | 8835 | 0.00 |

## Missing values (after cleaning)
| column | missing | present | missing_pct |
| --- | --- | --- | --- |
| record_id | 0 | 5678 | 0.00 |
| brand | 0 | 5678 | 0.00 |
| model | 0 | 5678 | 0.00 |
| processor | 0 | 5678 | 0.00 |
| ram_gb | 0 | 5678 | 0.00 |
| storage_gb | 2 | 5676 | 0.04 |
| storage_type | 0 | 5678 | 0.00 |
| age_years | 5678 | 0 | 100.00 |
| condition | 0 | 5678 | 0.00 |
| battery_health | 5678 | 0 | 100.00 |
| price | 218 | 5460 | 3.84 |
| currency | 0 | 5678 | 0.00 |
| source | 0 | 5678 | 0.00 |
| collection_date | 0 | 5678 | 0.00 |
| price_type | 0 | 5678 | 0.00 |
| notes | 0 | 5678 | 0.00 |
| price_lsl | 218 | 5460 | 3.84 |
| fx_rate_used | 0 | 5678 | 0.00 |
| fx_rate_date | 0 | 5678 | 0.00 |
| fx_rate_source | 0 | 5678 | 0.00 |
| quality_flags | 0 | 5678 | 0.00 |
| price_band | 218 | 5460 | 3.84 |
| is_duplicate | 0 | 5678 | 0.00 |

## Quality flags
| flag | records |
| --- | --- |
| missing_condition | 3934 |
| condition_unmapped | 3934 |
| price_outlier | 240 |
| missing_price | 218 |
| missing_storage_gb | 2 |

## Currency conversion
| currency | records | converted |
| --- | --- | --- |
| DZD | 5678 | 5460 |

## Split sizes
| split | records |
| --- | --- |
| test | 1236 |
| train | 3455 |
| val | 987 |

## Source breakdown
| source | records |
| --- | --- |
| Public dataset: bitsnaps/laptops_specs (Hugging Face, Apache-2.0) | 5678 |

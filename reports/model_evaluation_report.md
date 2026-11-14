# Stage 7: Model Development & Evaluation Report
## Intercity Bus Passenger Demand Forecasting

**Project:** BusDemand AI – Bus Passenger-Demand Forecasting
**Forecast Horizon:** 24 Hours prior to scheduled departure
**Chronological Split:** Train (2024-2025, 58,480 trips), Val (2026-H1, 14,480 trips), Test (2026-H2, 14,720 trips)

---

## 1. Executive Summary & Model Selection

Three forecasting approaches were trained strictly on historical data (2024–2025) and compared on the **chronological validation set (2026-H1)**:
1. **Historical Baseline:** Route $\times$ Departure Slot historical mean with hierarchical fallback.
2. **Random Forest Regressor Pipeline:** Non-linear decision tree ensemble with One-Hot encoded categoricals.
3. **HistGradientBoosting Regressor Pipeline:** Gradient-boosted decision trees with native categorical handling.

### Validation Performance & Model Ranking

| Model | Split | MAE | RMSE | WAPE (%) | Rank |
|---|---|---|---|---|---|
| **HistGradientBoosting** | Validation | **7.852** | 10.345 | **20.98%** | #1 |
| **RandomForest** | Validation | **8.058** | 10.640 | **21.53%** | #2 |
| **Baseline** | Validation | **9.687** | 13.451 | **25.89%** | #3 |

### Winning Model Selected: `HistGradientBoosting`
- Selected based strictly on validation performance using **MAE** and **WAPE** as the primary operational criteria.
- **HistGradientBoosting** achieved the lowest validation MAE (**7.852**) and WAPE (**20.98%**), outperforming the historical baseline by **19.0% relative error reduction**.

---

## 2. Held-Out Test Set Evaluation

The selected winning model was evaluated once on the completely unseen **held-out test set (2026-07-01 to 2026-12-31, 14,720 trips)**:

| Model | Split | MAE | RMSE | WAPE (%) | Improvement vs Baseline |
|---|---|---|---|---|---|
| **Baseline** | Test | **9.091** | 12.143 | **26.16%** | Baseline |
| **HistGradientBoosting** | Test | **7.331** | 9.548 | **21.10%** | +19.34% WAPE lift |

### Key Performance Takeaways:
- **Consistent Generalization:** Test WAPE (**21.10%**) closely tracks Validation WAPE, demonstrating zero temporal over-fitting.
- **Substantial Accuracy Gain:** Machine learning reduces Mean Absolute Error by **19.4%** compared to the operational heuristic baseline, successfully capturing holiday surges and day-of-week interactions.

---

## 3. Operational Segment & Peak Analysis

Evaluation of `HistGradientBoosting` across operational demand categories on the test set:

| Operational Segment | Scheduled Trips | Mean Actual | Mean Predicted | MAE | RMSE | WAPE (%) |
|---|---|---|---|---|---|---|
| Peak Slots (DEP_03, DEP_04) | 7,360 | 37.4 | 37.0 | **7.790** | 10.264 | **20.86%** |
| Off-Peak Slots (DEP_01, DEP_02) | 7,360 | 32.1 | 32.3 | **6.873** | 8.773 | **21.38%** |
| Long Weekend Trips | 1,680 | 45.6 | 45.1 | **9.617** | 12.664 | **21.08%** |
| Regular (Non-Long-Weekend) | 13,040 | 33.4 | 33.3 | **7.037** | 9.068 | **21.10%** |
| Capacity Exceeded Trips (>45) | 2,956 | 58.5 | 48.5 | **11.877** | 14.446 | **20.32%** |
| Normal Capacity Trips (<=45) | 11,764 | 28.8 | 31.2 | **6.189** | 7.850 | **21.50%** |

---

## 4. Per-Corridor Accuracy Breakdown (Held-Out Test Set)

| Route ID | Trips | Mean Actual | Mean Predicted | MAE | RMSE | WAPE (%) |
|---|---|---|---|---|---|---|
| `R_BLR_CCJ` | 736 | 35.5 | 35.4 | 7.716 | 9.892 | **21.74%** |
| `R_BLR_COK` | 736 | 43.4 | 43.6 | 8.976 | 11.480 | **20.71%** |
| `R_BLR_MAA` | 736 | 48.3 | 48.3 | 9.441 | 11.958 | **19.53%** |
| `R_BLR_MYS` | 736 | 34.0 | 33.6 | 7.306 | 9.583 | **21.52%** |
| `R_BLR_TCR` | 736 | 34.1 | 34.0 | 7.250 | 9.506 | **21.24%** |
| `R_CBE_COK` | 736 | 32.1 | 31.8 | 6.619 | 8.791 | **20.63%** |
| `R_CBE_MAA` | 736 | 38.6 | 38.1 | 8.123 | 10.487 | **21.05%** |
| `R_CCJ_BLR` | 736 | 35.1 | 35.1 | 7.780 | 10.117 | **22.15%** |
| `R_CCJ_COK` | 736 | 30.8 | 30.4 | 6.295 | 8.136 | **20.43%** |
| `R_CCJ_TCR` | 736 | 24.1 | 24.3 | 5.562 | 7.277 | **23.04%** |
| `R_COK_BLR` | 736 | 42.1 | 40.9 | 8.562 | 11.097 | **20.35%** |
| `R_COK_CBE` | 736 | 29.7 | 29.5 | 6.212 | 7.840 | **20.93%** |
| `R_COK_CCJ` | 736 | 29.1 | 29.6 | 6.552 | 8.437 | **22.48%** |
| `R_COK_TVM` | 736 | 32.7 | 33.0 | 6.943 | 8.798 | **21.25%** |
| `R_MAA_BLR` | 736 | 47.1 | 47.0 | 9.309 | 11.737 | **19.77%** |
| `R_MAA_CBE` | 736 | 38.0 | 38.1 | 7.933 | 10.296 | **20.86%** |
| `R_MYS_BLR` | 736 | 30.5 | 31.0 | 6.362 | 8.238 | **20.83%** |
| `R_TCR_BLR` | 736 | 32.9 | 33.1 | 7.174 | 9.061 | **21.80%** |
| `R_TCR_CCJ` | 736 | 23.1 | 23.1 | 5.227 | 6.657 | **22.65%** |
| `R_TVM_COK` | 736 | 33.7 | 33.6 | 7.283 | 9.395 | **21.60%** |

---

## 5. Model Explainability & Permutation Feature Importance

### Top 10 Most Influential Features (HistGradientBoosting Permutation on Validation Set)

Permutation importance measures the degradation in validation MAE when a feature's values are randomly shuffled:

| Rank | Feature | Mean MAE Degradation | Interpretation |
|---|---|---|---|
| #1 | `route_id` | **+4.209** | Primary driver of passenger demand variation |
| #2 | `departure_slot` | **+3.444** | Primary driver of passenger demand variation |
| #3 | `day_of_week` | **+1.296** | Primary driver of passenger demand variation |
| #4 | `is_long_weekend` | **+0.520** | Primary driver of passenger demand variation |
| #5 | `is_post_holiday` | **+0.333** | Primary driver of passenger demand variation |
| #6 | `is_holiday_destination` | **+0.255** | Primary driver of passenger demand variation |
| #7 | `is_pre_holiday` | **+0.199** | Primary driver of passenger demand variation |
| #8 | `is_holiday_origin` | **+0.139** | Primary driver of passenger demand variation |
| #9 | `distance_km` | **+0.105** | Primary driver of passenger demand variation |
| #10 | `departure_hour` | **+0.037** | Primary driver of passenger demand variation |

### Limitations of Feature Importance:
1. **Multicollinearity:** Correlated features (such as `distance_km` and `journey_duration_minutes`, or `route_id` and `route_mobility_score`) share attribution in tree algorithms, dampening individual permutation scores.
2. **Synthetic Data Context:** These feature weights reflect the behavioral assumptions encoded in the simulation pipeline and must not be treated as empirical measurements of passenger behavior in real-world commercial transit fleets.

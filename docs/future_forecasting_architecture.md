# BusDemand AI – Future Forecasting Data Architecture & Calendar Separation

## 1. Architectural Overview & Design Rationale

In operational bus demand forecasting, maintaining strict demarcation between **historical training data** and **future forecasting horizons** is essential to prevent data leakage, guarantee reproducibility, and maintain statistical integrity.

```
+-----------------------------------------------------------------------------------+
|                                BusDemand AI System                                |
+-----------------------------------------+-----------------------------------------+
                                          |
        +---------------------------------+---------------------------------+
        |                                                                   |
        v                                                                   v
+-----------------------------------+             +-----------------------------------+
|     Historical ML Pipeline        |             |    Operational Inference API      |
|          (2024 - 2026)            |             |          (FastAPI & Web UI)       |
+-----------------------------------+             +-----------------------------------+
| - data/processed/trip_demand.csv  |             | - Serves real-time predictions    |
|   (87,680 trips, 3 full years)    |             | - Enforces 24-hour advance cutoff |
| - data/raw/calendar_features.csv  |             | - Dynamic calendar selection:     |
|   (1,096 days, 2024-01-01 to      |             |     * 2024-2026 -> 'historical'   |
|    2026-12-31)                    |             |     * 2027      -> 'future'       |
| - Fixed chronological splits:     |             | - Emits identical 25-feature rows |
|     * Train: 2024 - 2025 (67%)    |             | - Loads immutable ML pipeline     |
|     * Val:   2026-H1     (17%)    |             |   (artifacts/best_model_          |
|     * Test:  2026-H2     (17%)    |             |    pipeline.joblib)               |
| - HistGradientBoosting model      |             +-----------------+-----------------+
+-----------------+-----------------+                               |
                  |                                                 |
                  v                                                 v
        [ Trained Artifacts ]                     +-----------------------------------+
        artifacts/best_model_                     |      Future Calendar Features     |
        pipeline.joblib                           |               (2027)              |
                                                  +-----------------------------------+
                                                  | - data/future/2027_calendar_      |
                                                  |   features.csv (365 days)         |
                                                  | - Pure calendar features only     |
                                                  | - Zero synthetic demand labels    |
                                                  | - Zero model retraining           |
                                                  +-----------------------------------+
```

### Key Architectural Decisions:
1. **Preservation of Historical Ground Truth**:
   The historical synthetic trip dataset (`data/processed/trip_demand.csv`, 87,680 trips) and the historical calendar (`data/raw/calendar_features.csv`, 1,096 days) remain strictly immutable. No future dates (2027) are appended to historical datasets, preventing pollution of training distributions and evaluation metrics.
2. **Dedicated Future Calendar Partition**:
   Calendar features for 2027 are isolated in `data/future/2027_calendar_features.csv`. This file contains exactly 365 daily rows spanning `2027-01-01` to `2027-12-31` with identical column schemas (27 calendar features) to those in the historical calendar.
3. **Zero Retraining Requirement**:
   The trained `HistGradientBoostingRegressor` pipeline relies on 25 contextual features derived from route topology, city profiles, and calendar dynamics. Because calendar features for 2027 match the exact column names, data types, and ordering expected by the preprocessor, the model generates forward inferences without retraining.
4. **Dynamic Calendar Routing**:
   The `FeatureBuilder` inspects the requested `departure_date`:
   - If `2024-01-01 <= departure_date <= 2026-12-31`: routed to `historical`.
   - If `2027-01-01 <= departure_date <= 2027-12-31`: routed to `future`.
   - Outside these bounds: rejected with an informative `DateOutOfRangeError`.

---

## 2. Dataset Separation & Immutability Guarantees

| Dataset Layer | File Path | Temporal Bounds | Records | Contents | Mutability Status |
|---|---|---|---|---|---|
| **Historical Demand** | `data/processed/trip_demand.csv` | 2024-01-01 to 2026-12-31 | 87,680 trips | Target demand (`passenger_demand`), operational capacity, route & calendar joins | **Immutable** (Zero 2027 rows added) |
| **Historical Calendar** | `data/raw/calendar_features.csv` | 2024-01-01 to 2026-12-31 | 1,096 days | 27 temporal & holiday columns | **Immutable** |
| **Future Calendar** | `data/future/2027_calendar_features.csv` | 2027-01-01 to 2027-12-31 | 365 days | 27 temporal & projected holiday columns | **Active Reference** |
| **Route Profiles** | `data/raw/route_profiles.csv` | Static network | 20 corridors | Corridors, distances, durations, mobility scores | **Immutable** |
| **City Profiles** | `data/raw/city_profiles.csv` | Static network | 8 urban hubs | Coordinates, tier, population weights | **Immutable** |

---

## 3. Projected 2027 Calendar & Holiday Classification

To maintain high forecasting precision while remaining transparent about projection uncertainty, 2027 holiday events are categorized into three explicit verification tiers:

### Verification Tiers:
1. `projected_statutory_holiday`:
   Fixed-date statutory national and state observances established by government legislation.
   - Republic Day (`2027-01-26`)
   - May Day / Labour Day (`2027-05-01`)
   - Independence Day (`2027-08-15`)
   - Gandhi Jayanti (`2027-10-02`)
   - Karnataka Rajyotsava (`2027-11-01`)
   - Christmas (`2027-12-25`)
2. `projected_astronomical_calendar`:
   Calculated from standard astronomical and Hindu Panchangam / Christian liturgical calendars.
   - Makara Sankranti / Pongal (`2027-01-15`)
   - Good Friday (`2027-03-26`)
   - Ugadi / Telugu New Year (`2027-04-07`)
   - Vishu (`2027-04-14`)
   - Onam / Thiruvonam (`2027-09-12`)
   - Deepavali (`2027-10-29`)
3. `uncertain_lunar_observation`:
   Islamic Hijri lunar calendar observances subject to regional moon-sighting committees and state gazette notification.
   - Id-ul-Fitr / Ramzan (`2027-03-10` projected)
   - Bakrid / Eid al-Adha (`2027-05-17` projected)
   - Muharram (`2027-06-16` projected)
   - Milad-un-Nabi (`2027-08-16` projected)

---

## 4. Operational Advance Cutoff & Date Validation

BusDemand AI enforces a strict **24-hour advance forecast rule** based on server time in the operating timezone (`Asia/Kolkata`):

$$\text{earliest\_departure\_datetime} = \text{current\_server\_time} + 24\text{ hours}$$

### Validation Rules:
1. **Past Departures**:
   Departures earlier than current server time are rejected (`400 Bad Request` with `OperationalCutoffError`).
2. **Sub-24-Hour Departures**:
   Departures less than 24 hours ahead of server time are rejected (`400 Bad Request` with `OperationalCutoffError`).
3. **Horizon Boundaries**:
   Departures beyond `2027-12-31` are rejected (`400 Bad Request` with `DateOutOfRangeError`).
4. **Historical Testing Bypass**:
   For reproducible offline evaluation against historical data, the `/predict` endpoint supports an internal `allow_historical=True` flag, which is disabled by default in operational production mode.

---

## 5. Methodological Notice: Out-of-Sample Forward Inference

> [!IMPORTANT]
> **Forward Forecast Disclaimer**:
> Forecasts generated for future calendar dates in 2027 represent **out-of-sample forward projections**.
> - The predictions are produced by applying the learned regression relationships (from the 2024–2026 historical training set) to projected calendar and calendar-derived features.
> - **Zero future passenger demand numbers have been fabricated or synthesized as ground truth**.
> - Actual future passenger demand will depend on real-world economic, operational, and exogenous conditions that may diverge from historical simulation distributions.

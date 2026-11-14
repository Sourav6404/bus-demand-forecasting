# Stage 6: Exploratory Data Analysis Report
## Synthetic Intercity Passenger Demand Simulation

**Project:** BusDemand AI – Bus Passenger-Demand Forecasting  
**Simulation Period:** 2024-01-01 to 2026-12-31 (1,096 calendar days)  
**Total Scheduled Trips:** 87,680  
**Dataset Path:** `data/processed/trip_demand.csv`  

---

## 1. Executive Summary & Descriptive Statistics

Passenger demand is simulated for 20 directed corridors across 4 daily departure slots using an extended spatial interaction Gravity Model and Negative Binomial stochastic sampling (dispersion parameter $\alpha = 0.04$, seed 42). Physical coach capacity is standardized to **45 seats**.

| Metric | Value | Operational Interpretation |
|---|---|---|
| **Total Trips** | **87,680** | 20 routes $\times$ 1,096 days $\times$ 4 departures |
| **Mean Passenger Demand** | **35.76 passengers** | Average simulated load factor of **79.5%** |
| **Standard Deviation** | **15.70** | Robust behavioral spread across corridors |
| **Median (IQR)** | **33 (25 – 44)** | 50% of trips fall in steady commercial range |
| **Minimum Demand** | **2 passengers** | Quiet midweek/monsoon off-peak slots |
| **95th Percentile** | **65 passengers** | Severe capacity strain on prime services |
| **Maximum Demand** | **195 passengers** | Extreme festive booking surge (requested demand, not physical occupancy) |
| **Capacity-Exceeded Trips** | **19,348 (22.07%)** | Result of synthetic scenario simulation (not an empirical industry benchmark) |
| **Total Unserved Demand** | **263,766 passengers** | Latent unmet passenger volume (waitlists/spillover) |

> [!NOTE]
> **Synthetic Simulation Notice:** The 22.07% capacity-exceeded rate and 195-passenger maximum demand are specific outcomes of the synthetic gravity model, holiday multipliers, and stochastic parameters chosen for this benchmark. They represent unconstrained passenger requested demand rather than the physical seating capacity of a 45-seat bus.

---

## 2. Demand by Departure Slot

Intercity travel demonstrates sharp time-of-day differentiation driven by journey length:

| Slot ID | Departure Time | Description | Mean Demand | Std Dev | Median | % Exceeded (>45) |
|---|---|---|---|---|---|---|
| `DEP_01` | 01 | Early Morning (06:00) | **37.98** | 13.09 | 36 | **24.46%** |
| `DEP_02` | 02 | Midday (11:30) | **28.59** | 10.19 | 27 | **6.37%** |
| `DEP_03` | 03 | Evening (17:00) | **39.41** | 14.49 | 37 | **27.73%** |
| `DEP_04` | 04 | Night Sleeper (22:30) | **37.08** | 20.72 | 32 | **29.70%** |

> [!NOTE]
> **Key Finding:** `DEP_03` (Evening 17:00) and `DEP_01` (Early Morning 06:00) dominate short-to-medium regional corridors (e.g., Bengaluru $\leftrightarrow$ Mysuru), while `DEP_04` (Night Sleeper 22:30) exhibits the highest variance and maximum passenger surges (up to 195 passengers) on overnight corridors (e.g., Bengaluru $\leftrightarrow$ Kochi, Chennai).

---

## 3. Day of Week & Directional Commuter Flow

Weekly patterns reflect metropolitan workforce migration between Tier 1 economic centers (Bengaluru, Chennai) and regional urban centers:

| Day of Week | Scheduled Trips | Mean Demand | Std Dev | Median | % Exceeded (>45) |
|---|---|---|---|---|---|
| **Monday** | 12,560 | **34.20** | 14.32 | 32 | 18.80% |
| **Tuesday** | 12,560 | **31.75** | 12.79 | 30 | 14.19% |
| **Wednesday** | 12,560 | **32.16** | 13.09 | 30 | 14.90% |
| **Thursday** | 12,560 | **34.52** | 13.95 | 33 | 19.16% |
| **Friday** | 12,480 | **41.64** | 19.40 | 38 | 33.73% |
| **Saturday** | 12,480 | **37.03** | 14.90 | 35 | 25.20% |
| **Sunday** | 12,480 | **39.09** | 17.71 | 36 | 28.61% |

### Behavioral Insights:
- **Weekend vs Weekday:** Weekend mean (**38.06**) exceeds weekday mean (**34.85**) by **+9.2%**.
- **Friday Surge:** Friday experiences the highest departure volume as workforce and students depart Tier 1 hubs for hometowns.
- **Sunday Inbound:** Sunday evening and night departures inbound to Bengaluru and Chennai experience peak return demand with over **30%** of trips exceeding coach capacity.
- **Midweek Lull:** Tuesday and Wednesday exhibit baseline demand (31.75 passengers).

---

## 4. Seasonal Dynamics (IMD Classifications)

| Season | Calendar Months | Trips | Mean Demand | Std Dev | % Exceeded (>45) |
|---|---|---|---|---|---|
| **Winter** | 14,240 | **35.77** | 15.35 | 21.59% |
| **Summer** | 22,080 | **39.26** | 16.73 | 29.32% |
| **Monsoon** | 29,280 | **31.61** | 13.78 | 13.97% |
| **Post-Monsoon** | 22,080 | **37.77** | 16.00 | 25.85% |

### Seasonal Observations:
- **Post-Monsoon (Oct–Dec):** Peak transit season (**37.77**) driven by consecutive festive cycles (Dussehra, Diwali, Christmas) and optimal travel climate.
- **Summer (Mar–May):** Strong family and student holiday travel (**39.26**).
- **Monsoon (Jun–Sep):** Lowest demand (**31.61**), where South-West monsoon weather significantly dampens Kerala highway mobility outside the Onam festival window.

---

## 5. Holiday & Long-Weekend Impact

| Travel Condition | Segment | Mean Demand | Lift vs Baseline |
|---|---|---|---|
| **Long Weekend** | `is_long_weekend == True` | **47.27** | **+36.6%** |
| **Regular Days** | `is_long_weekend == False` | 34.61 | Baseline |
| **National Holidays** | `is_national_holiday == True` | **50.04** | **+40.4%** |
| **Pre-Holiday Eve** | `is_pre_holiday == True` | **41.66** | Outbound surge |
| **Post-Holiday Return**| `is_post_holiday == True` | **43.72** | Inbound return |

---

## 6. Directional Corridor Comparisons (All 10 Bidirectional Pairs)

| Corridor | Forward Route (Mean) | Reverse Route (Mean) | Directional Difference |
|---|---|---|---|
| Bengaluru <-> Chennai | `R_BLR_MAA`: **49.84** (53.8%) | `R_MAA_BLR`: **48.09** (49.5%) | **+1.75** |
| Bengaluru <-> Kochi | `R_BLR_COK`: **45.13** (41.5%) | `R_COK_BLR`: **42.39** (35.4%) | **+2.74** |
| Bengaluru <-> Kozhikode | `R_BLR_CCJ`: **36.77** (22.7%) | `R_CCJ_BLR`: **36.19** (21.3%) | **+0.58** |
| Bengaluru <-> Thrissur | `R_BLR_TCR`: **35.29** (19.6%) | `R_TCR_BLR`: **34.14** (17.7%) | **+1.16** |
| Bengaluru <-> Mysuru | `R_BLR_MYS`: **34.50** (18.6%) | `R_MYS_BLR`: **31.41** (12.7%) | **+3.08** |
| Chennai <-> Coimbatore | `R_MAA_CBE`: **39.14** (28.0%) | `R_CBE_MAA`: **39.14** (27.7%) | **+0.00** |
| Coimbatore <-> Kochi | `R_CBE_COK`: **33.18** (15.6%) | `R_COK_CBE`: **30.58** (10.7%) | **+2.60** |
| Kochi <-> Thiruvananthapuram | `R_COK_TVM`: **34.10** (18.1%) | `R_TVM_COK`: **34.71** (18.5%) | **+-0.60** |
| Kochi <-> Kozhikode | `R_COK_CCJ`: **30.40** (10.5%) | `R_CCJ_COK`: **31.51** (12.9%) | **+-1.11** |
| Kozhikode <-> Thrissur | `R_CCJ_TCR`: **25.04** (4.2%) | `R_TCR_CCJ`: **23.70** (2.3%) | **+1.34** |

---

## 7. Outliers & Peak Festive Surges

Trips with passenger demand exceeding **120 passengers** occur exclusively during documented peak holiday conjunctions:
1. **Diwali Eve (October/November):** Major exodus across Bengaluru $\to$ Chennai, Kochi, Kozhikode.
2. **Onam Eve (Late August / September):** Massive surge on Bengaluru $\to$ Kochi (`R_BLR_COK`) and Bengaluru $\to$ Kozhikode (`R_BLR_CCJ`).
3. **Pongal (Mid-January):** Peak outbound flows on Chennai $\to$ Coimbatore (`R_MAA_CBE`) and Bengaluru $\to$ Chennai (`R_BLR_MAA`).
4. **Christmas / New Year (Late December):** Tourism surges towards Kochi and Mysuru.

These peak surges generate realistic unserved demand (waitlists) without distorting normal operational baselines. Note that 195 passengers represents simulated booking requests, not physical bus occupancy (which is strictly capped at 45 seats).

---

## 8. Forecast Horizon, Metadata & Machine Learning Usage

### A. 24-Hour Forecast Horizon & Prediction Cutoff
- **Cutoff Definition:** $T_{	ext{departure}} - 24.0	ext{ hours}$.
- **Information Availability Assumption:** The forecasting cutoff assumes that schedule timetables, calendar structures, and planned gazetted holiday dates are available 24 hours prior to departure.
- **Uncertain / Moon-Sighting Dates:** In real-world transit operations, certain holiday dates (such as Eid-ul-Fitr, Bakrid, and Muharram) depend on lunar sightings confirmed closer to the event. In this benchmark dataset, planned gazetted dates are treated as known scenario reference points available at the prediction cutoff.

### B. Feature Usage & Metadata Distinction
- **`trip_id`:** This is an operational composite primary key (e.g. `TRIP_R_BLR_COK_20241031_DEP_04`). **It must NOT be passed directly as a feature into machine learning models.**
- **Categorical Variables:** Corridor identifiers (`route_id`), terminal cities (`origin_city_id`, `destination_city_id`), and schedule windows (`departure_slot`) must be handled as categorical variables through appropriate encoding techniques (e.g., Target Encoding, One-Hot Encoding, or native categorical handles).
- **Target Segregation:** `passenger_demand` is the sole prediction target. Operational fields (`capacity_exceeded` and `estimated_unserved_demand`) are post-prediction evaluation metrics and must be excluded from feature matrices to prevent target leakage.

# Synthetic Data Strategy & Behavioral Demand Assumptions

This document outlines the theoretical foundations, behavioral assumptions, and simulation methodology for generating high-resolution synthetic intercity bus passenger demand in **BusDemand AI**.

---

## 1. Motivation & Scope

Real-world intercity transit ticketing data is often proprietary, siloed across multiple private operators (KSRTC, SETC, KSRTC-Kerala, private fleet aggregates), and subject to ticketing irregularities. To establish a rigorous, reproducible modeling benchmark, BusDemand AI implements a behavioral synthetic data generation pipeline rooted in transport economics, spatial interaction models, and empirical passenger choice theory.

### Core Simulation Parameters
| Parameter | Value | Description |
|---|---|---|
| **Horizon** | 3 Years (2024-01-01 to 2026-12-31) | 1,096 continuous simulation days |
| **Geographic Focus** | 8 South Indian Urban Hubs | Bengaluru, Mysuru, Chennai, Coimbatore, Kochi, Kozhikode, Thrissur, Thiruvananthapuram |
| **Corridors** | Pairwise Intercity Routes | Directed Origin-Destination (OD) pairs connecting major economic, educational, and transit hubs |
| **Departures** | 4 daily departure slots per route | Early Morning (06:00), Midday (11:30), Evening (17:00), Night Sleeper (22:30) |
| **Random Seed** | `42` (Fixed & deterministic) | Full reproducibility across all random processes |

---

## 2. Gravity Model Foundation for Spatial Interaction

Intercity travel volume between city $i$ (origin) and city $j$ (destination) is governed by an extended spatial interaction gravity model:

$$T_{ij} \propto \frac{G_i^{\alpha} \cdot A_j^{\beta}}{d_{ij}^{\gamma}} \cdot M_{ij}$$

Where:
- $G_i$ = **Demand Generation Index** of Origin City $i$ (propensity of residents and workforce to travel outbound).
- $A_j$ = **Demand Attraction Index** of Destination City $j$ (commercial, educational, tourism, and healthcare draw).
- $d_{ij}$ = Highway road distance between city $i$ and city $j$ in kilometers.
- $\gamma$ = Distance friction parameter ($\approx 1.1 - 1.4$ for highway bus travel in South India, where distances under 150 km face high private transit/train competition, 150–500 km is the prime bus market, and >600 km competes with aviation/sleeper trains).
- $M_{ij}$ = Corridor mobility coupling index (state-border friction, highway corridor quality, cultural and economic interdependence e.g., Bengaluru–Chennai or Bengaluru–Kochi).

---

## 3. Socioeconomic City Profile Indices

Each of the 8 cities is assigned synthetic component scores normalized on a **0.0 to 1.0 scale** across six core dimensions. These scores represent controlled scenario modelling assumptions:

1. **Urban Scale ($S_u$):** Relative population concentration and metropolitan scale.
2. **Employment Score ($S_{emp}$):** Concentration of IT, manufacturing, industrial corridors, and corporate workforce.
3. **Education Score ($S_{edu}$):** Density of premier universities, colleges, and student migration catchment.
4. **Tourism Score ($S_{tour}$):** Heritage, religious, pilgrimage, and leisure visitor attractions.
5. **Economic Activity Score ($S_{econ}$):** Commercial banking, wholesale trade, retail, and regional enterprise activity.
6. **Migration Score ($S_{mig}$):** Interstate and interdistrict workforce and student mobility flux.

### Derived Indices & Weight Formulation
- **Demand Generation Index ($DGI$):**
  $$DGI = 0.10 \cdot S_u + 0.25 \cdot S_{emp} + 0.20 \cdot S_{edu} + 0.05 \cdot S_{tour} + 0.15 \cdot S_{econ} + 0.25 \cdot S_{mig}$$
  *Rationale:* Generation places higher weight on employment, education, and migration as primary drivers of outbound intercity trips.

- **Demand Attraction Index ($DAI$):**
  $$DAI = 0.10 \cdot S_u + 0.20 \cdot S_{emp} + 0.10 \cdot S_{edu} + 0.25 \cdot S_{tour} + 0.25 \cdot S_{econ} + 0.10 \cdot S_{mig}$$
  *Rationale:* Attraction places higher weight on commercial activity, tourism, and employment opportunities as destination draws.

- **City Mobility Index ($CMI$):**
  $$CMI = \frac{DGI + DAI}{2}$$
  *Rationale:* Provides a balanced summary indicator of total corridor mobility participation.

---

## 4. Directed Route Network Topology & Corridor Modelling

Stage 4 establishes a directed intercity bus route network comprising **20 directed routes** spanning 10 strategic bidirectional corridors between the 8 South Indian urban hubs.

> [!IMPORTANT]
> **Synthetic Modelling Assumptions Notice:** All route distances, journey durations, commercial speeds, and route demand multipliers in this project are synthetic modelling assumptions constructed for prototype demand simulation. They do **not** represent official measurements, certified transit schedules, or empirical telemetry from road transport authorities (such as NHAI, KSRTC, SETC, or private operators).

### A. Directed Asymmetry & Static Multiplier Definition
A route from city $A \to B$ and its reverse $B \to A$ are treated as independent directed routes with separate `route_id`s, independent tracking, and distinct `route_demand_multiplier`s:
- **Static Scenario Adjustment:** The `route_demand_multiplier` is a **static, direction-specific baseline scenario calibration factor** that captures structural corridor characteristics (such as baseline commercial exchange density, highway corridor quality, or established migrant catchment size).
- **Strict Separation from Temporal Effects:** The `route_demand_multiplier` does **not** model temporal variations such as Friday/Sunday weekend spikes, day-of-week cycles, holiday surges, or festival effects. All temporal, calendar, and seasonal dynamics are modelled dynamically as separate time-varying multipliers in subsequent simulation stages.
- **Directional Baseline Calibration:** Corridors originating in high-employment Tier 1 metropolises (e.g., Bengaluru $\to$ Kochi: 1.28, Bengaluru $\to$ Chennai: 1.25) exhibit elevated baseline multipliers reflecting structural outbound workforce and student mobility, while reverse journeys (e.g., Kochi $\to$ Bengaluru: 1.25, Chennai $\to$ Bengaluru: 1.22) reflect steady baseline return flows.

### B. Route Features & Formulations
1. **Distance ($d_{ij}$):** Synthetic road distance in kilometers based on illustrative highway alignments (NH 48, NH 544, NH 66, NH 766).
2. **Estimated Duration ($t_{ij}$):** Synthetic intercity bus journey time in hours, incorporating illustrative urban terminal clearance, passenger stops, and terrain friction.
3. **Average Commercial Speed ($v_{ij}$):**
   $$v_{ij} = \frac{d_{ij}}{t_{ij}}$$
   *Synthetic speed assumptions range between 37.5 km/h and 53.7 km/h:*
   - Flat expressway corridors (e.g., Chennai $\leftrightarrow$ Coimbatore: ~53.7 km/h, Bengaluru $\leftrightarrow$ Chennai: ~53.1 km/h).
   - Ghat sections and hilly terrain (e.g., Bengaluru $\leftrightarrow$ Kozhikode via Wayanad: ~45.0 km/h).
   - Dense ribbon-development coastal highways (e.g., Kochi $\leftrightarrow$ Kozhikode: ~37.5 km/h, Kochi $\leftrightarrow$ Thiruvananthapuram: ~38.2 km/h).
4. **Origin Generation Index ($DGI_o$) & Destination Attraction Index ($DAI_d$):** Directionally mapped from the base city profile table.
5. **Route Mobility Score ($RMS_{ij}$):**
   $$RMS_{ij} = \frac{DGI_o + DAI_d}{2}$$
   Represents a combined linear summary score of the corridor's bilateral mobility potential.
6. **Fleet Bus Capacity:** Standardized to 45 seats per departure coach.

### C. Role in Upcoming Passenger-Demand Formula & Avoiding Double-Counting
In the upcoming behavioral passenger-demand simulation stage, daily departure-level demand will be generated using an extended Gravity Model formulation:

$$\text{BaseDemand}_{ij} \propto f(\text{Mobility}_{ij}) \cdot \frac{1}{d_{ij}^\gamma} \cdot \text{route\_demand\_multiplier}_{ij}$$

Where:
- **Avoiding Double-Counting of Mobility Indices:** The simulator must choose **either** the granular directional product $(DGI_o \cdot DAI_d)$ **or** the combined summary `route_mobility_score` ($RMS_{ij}$). **Under no circumstances should both terms be multiplied together.** Multiplying both would apply the city generation and attraction factors twice, artificially compounding metropolitan scale and distorting the gravity curve.
- **Role of `route_demand_multiplier`:** Modulates the baseline corridor passenger volume as a static exogenous weight ($M_{ij}$), ensuring corridor-specific trade and migration intensity is respected independently of city size and distance.
- **Separation of Concerns:** The static base demand is subsequently modulated by dynamic temporal modifiers (Day-of-Week multiplier, Seasonal/Festival surge multiplier, and Departure-slot preference) without conflating static corridor weights with temporal shocks.

### D. 20-Route Network Overview
*Note: All numerical values below are synthetic prototype scenario assumptions.*

| Route ID | Origin $\to$ Destination | Distance (km) | Duration (hrs) | Speed (km/h) | Mobility Score | Demand Multiplier | Scenario Rationale |
|---|---|---|---|---|---|---|---|
| `R_BLR_MYS` | Bengaluru $\to$ Mysuru | 145.0 | 3.0 | 48.33 | 0.7878 | 1.10 | Expressway weekend tourism and commuter corridor |
| `R_MYS_BLR` | Mysuru $\to$ Bengaluru | 145.0 | 3.0 | 48.33 | 0.7065 | 1.08 | Commuter and student return flow to state capital |
| `R_BLR_MAA` | Bengaluru $\to$ Chennai | 345.0 | 6.5 | 53.08 | 0.9095 | 1.25 | Metropolitan IT and manufacturing business corridor |
| `R_MAA_BLR` | Chennai $\to$ Bengaluru | 345.0 | 6.5 | 53.08 | 0.9018 | 1.22 | Metropolitan commercial and workforce reverse flow |
| `R_BLR_CCJ` | Bengaluru $\to$ Kozhikode | 360.0 | 8.0 | 45.00 | 0.8032 | 1.15 | Ghat corridor connecting Malabar diaspora and students |
| `R_CCJ_BLR` | Kozhikode $\to$ Bengaluru | 360.0 | 8.0 | 45.00 | 0.7785 | 1.12 | Malabar return travel to Bengaluru employment market |
| `R_BLR_TCR` | Bengaluru $\to$ Thrissur | 470.0 | 9.5 | 49.47 | 0.7900 | 1.12 | Long-distance overnight cultural and commercial connection |
| `R_TCR_BLR` | Thrissur $\to$ Bengaluru | 470.0 | 9.5 | 49.47 | 0.7488 | 1.10 | Central Kerala workforce return flow |
| `R_BLR_COK` | Bengaluru $\to$ Kochi | 540.0 | 11.0 | 49.09 | 0.8868 | 1.28 | High-volume tech workforce and tourism sleeper route |
| `R_COK_BLR` | Kochi $\to$ Bengaluru | 540.0 | 11.0 | 49.09 | 0.8230 | 1.25 | Kochi port and IT diaspora return flow to Bengaluru |
| `R_MAA_CBE` | Chennai $\to$ Coimbatore | 510.0 | 9.5 | 53.68 | 0.8212 | 1.18 | Industrial and educational connector between TN hubs |
| `R_CBE_MAA` | Coimbatore $\to$ Chennai | 510.0 | 9.5 | 53.68 | 0.8102 | 1.15 | Textile and engineering commercial travel to state capital |
| `R_CBE_COK` | Coimbatore $\to$ Kochi | 190.0 | 4.5 | 42.22 | 0.7875 | 1.05 | Palakkad Gap regional trade and airport feeder route |
| `R_COK_CBE` | Kochi $\to$ Coimbatore | 190.0 | 4.5 | 42.22 | 0.7425 | 1.02 | Regional commercial and industrial feeder route |
| `R_COK_CCJ` | Kochi $\to$ Kozhikode | 180.0 | 4.8 | 37.50 | 0.7120 | 1.06 | Coastal business, trade, and medical connection |
| `R_CCJ_COK` | Kozhikode $\to$ Kochi | 180.0 | 4.8 | 37.50 | 0.7510 | 1.04 | Malabar regional travel to commercial capital |
| `R_COK_TVM` | Kochi $\to$ Thiruvananthapuram | 210.0 | 5.5 | 38.18 | 0.7548 | 1.12 | Commercial to administrative and Technopark connector |
| `R_TVM_COK` | Thiruvananthapuram $\to$ Kochi | 210.0 | 5.5 | 38.18 | 0.7777 | 1.10 | Administrative capital to port city commercial return flow |
| `R_CCJ_TCR` | Kozhikode $\to$ Thrissur | 115.0 | 3.0 | 38.33 | 0.6542 | 0.95 | Regional transit connecting Malabar and cultural capital |
| `R_TCR_CCJ` | Thrissur $\to$ Kozhikode | 115.0 | 3.0 | 38.33 | 0.6378 | 0.92 | Regional trade and education feeder route |

### E. Assumptions & Limitations
- **Synthetic Modelling Assumptions:** Distances, journey durations, commercial speeds, and route demand multipliers are illustrative approximations designed for benchmark simulation rather than live GPS telemetry or certified road transport schedules.
- **Topological Boundary:** The 20 routes form a connected graph covering all 8 hubs; non-direct pairs (e.g., Mysuru $\leftrightarrow$ Thiruvananthapuram) are assumed to operate via multi-leg transfers or outside the core 20-corridor prototype scope.

---

## 5. Behavioral Demand Dynamics & Modifiers

Synthetic daily demand incorporates non-linear multiplicative factors representing real passenger behavior:

### A. Day-of-Week (DoW) Seasonality
- **Friday & Sunday Spikes:** Peak intercity travel occurs on Friday evening (weekend return to home towns) and Sunday evening / Monday morning (return to work centers like Bengaluru and Chennai).
- **Midweek Baseline:** Tuesday and Wednesday exhibit steady baseline commercial and essential travel.

### B. Calendar & Festive Surges
South Indian intercity travel experiences distinct surge periods:
- **Diwali / Deepavali (Oct/Nov):** Intense regional travel across all corridors.
- **Pongal / Sankranti (Mid-January):** Peak mobility in Tamil Nadu corridors (Chennai, Coimbatore).
- **Onam (August/September):** Heavy inbound migration to Kerala cities (Kochi, Kozhikode, Thrissur, Thiruvananthapuram) predominantly originating from Bengaluru and Chennai.
- **Christmas & New Year (Late December):** Tourism surges towards Kochi and Mysuru.
- **Summer Vacation (April–May):** Family and student travel surges.
- **Monsoon Dip (June–July):** Kerala and Western Ghats routes experience weather-induced transit dampening.

### C. Departure Slot Preferences
- **Early Morning (06:00):** Dominated by regional business commuters and short-haul travelers (e.g., Bengaluru $\leftrightarrow$ Mysuru, Coimbatore $\leftrightarrow$ Thrissur).
- **Midday (11:30):** Leisure, family, and non-time-critical trips.
- **Evening (17:00):** After-work intercity weekenders.
- **Night Sleeper (22:30):** High-yield long-distance corridor travel (e.g., Bengaluru $\leftrightarrow$ Kochi, Chennai $\leftrightarrow$ Thiruvananthapuram).

### D. Elasticity & Group Dynamics
- **Price Sensitivity:** Simulates passenger churn when fares exceed baseline corridor thresholds.
- **Group Bookings:** Poisson-distributed group arrivals (solo travelers vs couples vs families) introducing realistic integer dispersion and variance-to-mean ratios.

---

## 6. Calendar, Holiday & Seasonal Feature Architecture

Stage 5 establishes a reference temporal framework spanning the complete 3-year simulation horizon (**1,096 calendar dates: 2024-01-01 to 2026-12-31**). It generates two foundational datasets:
1. `data/raw/calendar_features.csv`: Daily calendar matrix with temporal, seasonal, holiday, and vacation flags.
2. `data/raw/holiday_events.csv`: Audit registry of 160 official public holiday events with state jurisdiction, gazette citation, source URL, and verification status.

### A. Official Holiday Research & Gazette Grounding
Holidays are strictly grounded in official state government notifications across the three network states:
- **Karnataka (KA):** Department of Personnel and Administrative Reforms (DPAR)
  - 2024: Notification No. DPAR 16 HHL 2023, dated 28.11.2023
  - 2025: Notification No. DPAR 16 HHL 2024, dated 21.11.2024
  - 2026: Notification No. DPAR 16 HHL 2025, dated 17.11.2025
- **Kerala (KL):** General Administration (Co-ordination) Department (GAD)
  - 2024: G.O. (P) No. 24/2023/GAD, dated 04.10.2023
  - 2025: G.O. (P) No. 34/2024/GAD & No. 35/2024/GAD, dated 14.10.2024
  - 2026: G.O. (P) No. 14/2025/GAD & No. 15/2025/GAD, dated 31.10.2025
- **Tamil Nadu (TN):** Public (Miscellaneous) Department
  - 2024: G.O. (Ms.) No. 692, dated 09.11.2023
  - 2025: G.O. (Ms.) No. 792, dated 22.11.2024
  - 2026: G.O. (Ms.) No. 708, dated 11.11.2025
- **National (ALL):** Ministry of Personnel, Public Grievances and Pensions (DoPT)
  - Republic Day (Jan 26), Independence Day (Aug 15), Mahatma Gandhi Jayanti (Oct 2)

*Rule on Islamic Observances:* Holidays subject to the visibility of the moon (Eid-ul-Fitr, Bakrid, Muharram, Milad-un-Nabi) preserve documented official notes reflecting lunar sighting conditions.

### B. Route-Calendar State Intersection
In the route network, each route $i \to j$ spans an origin state and a destination state:
- **Intrastate Corridors:** (e.g. `BLR` $\leftrightarrow$ `MYS` in KA, `MAA` $\leftrightarrow$ `CBE` in TN, `COK` $\leftrightarrow$ `CCJ` in KL) synchronize directly with their respective state holiday flag (`is_holiday_ka`, `is_holiday_tn`, or `is_holiday_kl`).
- **Interstate Corridors:** (e.g. `BLR` $\leftrightarrow$ `COK` connecting KA and KL, or `BLR` $\leftrightarrow$ `MAA` connecting KA and TN) observe both state calendars:
  - An origin state holiday stimulates early outbound departures (`is_pre_holiday`).
  - A destination state festival (such as Onam in Kerala or Pongal in Tamil Nadu) draws massive inbound passenger traffic originating from employment centers in Karnataka.
  - Conjoined long weekends (`is_long_weekend`) maximize intercity travel demand across state borders.

### C. Seasonal & Vacation Assumptions
- **Meteorological Seasons (IMD Standard):**
  - **Winter:** January – February
  - **Summer (Pre-Monsoon):** March – May
  - **Monsoon (South-West Monsoon):** June – September
  - **Post-Monsoon (North-East Monsoon):** October – December
- **Vacation Periods (Clearly Labeled Prototype Assumptions):**
  - **Karnataka Schools (`is_school_vacation_ka`):** Summer break (Apr 11 – May 28); Dasara break (Oct 1 – Oct 14).
  - **Kerala Schools (`is_school_vacation_kl`):** Summer vacation (Apr 1 – May 31); Onam holidays (~10 days late Aug / early Sep); Christmas break (Dec 23 – Jan 2).
  - **Tamil Nadu Schools (`is_school_vacation_tn`):** Summer break (Apr 21 – Jun 5); Pongal holidays (Jan 13 – Jan 18).
  - **Higher Education (`is_college_vacation`):** Summer semester recess (May 15 – Jul 10); Winter inter-semester break (Dec 22 – Jan 3).

---

## 7. Stage 6 – Synthetic Passenger Demand Simulation Architecture

Stage 6 produces the core trip-level forecasting benchmark dataset (`data/processed/trip_demand.csv`), comprising **87,680 records** across 20 directed corridors, 1,096 calendar dates, and 4 daily departures.

> [!IMPORTANT]
> **Synthetic Modelling Notice:** This dataset represents controlled scenario simulation constructed from transport economics and empirical choice principles. It does **not** contain proprietary passenger bookings, GPS telemetry, or live ticketing registers from state or private transit operators.

### A. Prediction Target, Forecast Horizon & Metadata Definition
- **Target Variable:** `passenger_demand` (Discrete, non-negative integer count).
  - *Definition:* The total number of passengers who attempted to reserve or requested travel on a specific scheduled bus departure.
  - *Unconstrained Requested Demand:* Demand is **not** artificially capped at coach capacity ($N = 45$). Demand and capacity are distinct physical concepts: `passenger_demand` reflects total simulated booking requests, whereas physical coach occupancy cannot exceed 45 seats. Extreme values (such as the maximum simulated demand of 195 passengers on peak festive sleeper departures) represent intense booking demand attempting to reserve a single bus, resulting in 45 seated passengers and 150 unserved/waitlisted requests.
- **Forecast Horizon & Prediction Cutoff:**
  - *Forecast Horizon:* **24 hours prior to scheduled departure**.
  - *Prediction Cutoff Time:* $T_{\text{departure}} - 24.0\text{ hours}$.
  - *Information Availability Assumption at Cutoff:* The cutoff assumes that calendar features, schedule timetables, and planned gazetted holiday dates are known in advance. However, in operational reality, certain holiday dates (such as Islamic observances like Eid-ul-Fitr and Bakrid) depend on lunar sightings confirmed closer to the date, and occasional ad-hoc government declarations occur. In this synthetic baseline, planned gazette dates serve as deterministic scenario features available at cutoff.
- **Feature Typing & Machine Learning Usage Guidelines:**
  - `trip_id` is a composite primary key / audit metadata string (e.g., `TRIP_R_BLR_COK_20241031_DEP_04`) designed strictly for indexing, joins, and logging. **It must NOT be passed directly as a feature into machine learning models.**
  - `route_id`, `origin_city_id`, `destination_city_id`, and `departure_slot` represent discrete entity identifiers and must be encoded appropriately as categorical variables (e.g., Target Encoding, One-Hot Encoding, or CatBoost/LightGBM native categorical handles).
- **Strict Target Segregation (No Target Leakage):**
  - **Prediction Features (Available at $T - 24\text{h}$):** Route characteristics, scheduled time, calendar dimensions, holiday indicators, vacation flags, physical coach capacity.
  - **Target & Operational Fields (Excluded from Model Inputs):** `passenger_demand`, `capacity_exceeded` ($= \text{demand} > 45$), and `estimated_unserved_demand` ($= \max(0, \text{demand} - 45)$).

### B. Mathematical Demand Generation Formulation
The expected passenger demand rate $\mu_{ijst}$ for corridor $i \to j$, departure slot $s$, on date $t$ is formulated as a multiplicative behavioral model:

$$\mu_{ijst} = \text{BaseRate}_{ij} \cdot M_{\text{slot}}(d_{ij}, s) \cdot M_{\text{dow}}(i, j, s, t) \cdot M_{\text{season}}(t, i, j) \cdot M_{\text{hol}}(t, i, j) \cdot M_{\text{vac}}(t, i, j)$$

Where:
1. **Base Rate (Avoiding Double-Counting):**
   $$\text{BaseRate}_{ij} = \mu_0 \cdot \text{RMS}_{ij} \cdot \text{RDM}_{ij}$$
   - Baseline scaling factor $\mu_0 = 37.5$ seats.
   - $\text{RMS}_{ij}$: Route mobility score (derived from origin generation and destination attraction indices).
   - $\text{RDM}_{ij}$: Direction-specific static baseline multiplier.
   - *Rule:* Granular $(DGI_o \cdot DAI_d)$ is **not** multiplied with $\text{RMS}_{ij}$, strictly adhering to the double-counting avoidance principle established in Section 4.C.

2. **Distance-Slot Interaction ($M_{\text{slot}}$):**
   Captures passenger departure time preferences as a function of corridor journey distance:
   - **Short Corridors ($d < 250$ km):** Morning business commuter peak (`DEP_01`, 06:00: $\times 1.18$) and evening return peak (`DEP_03`, 17:00: $\times 1.22$); off-peak midday (`DEP_02`: $\times 0.88$) and night (`DEP_04`: $\times 0.72$).
   - **Long Corridors ($d \ge 250$ km):** Overnight sleeper preference dominates (`DEP_04`, 22:30: $\times 1.28$); steady morning daytime (`DEP_01`: $\times 1.00$), evening (`DEP_03`: $\times 1.02$), and low midday (`DEP_02`: $\times 0.78$).

3. **Day-of-Week & Directional Commuter Flow ($M_{\text{dow}}$):**
   Models metropolitan workforce mobility centered on Tier 1 economic engines (Bengaluru `BLR` and Chennai `MAA`):
   - **Friday Outbound Surge:** Workforce departing BLR/MAA for hometowns: evening/night slots surge by $\times 1.35$.
   - **Sunday Return Rush:** Hometown travelers returning to BLR/MAA: evening/night slots surge by $\times 1.38$.
   - **Midweek Baseline:** Tuesday and Wednesday exhibit steady commercial baseline ($\times 0.90$).

4. **Seasonal Dynamics ($M_{\text{season}}$):**
   - Post-Monsoon (Oct–Dec): $\times 1.08$ (festive quarter, pleasant weather).
   - Summer (Mar–May): $\times 1.08$ (annual holiday mobility).
   - Winter (Jan–Feb): $\times 1.02$.
   - Monsoon (Jun–Sep): $\times 0.88$ for corridors serving Kerala (South-West monsoon road dampening); $\times 0.94$ elsewhere.
   - Month boundary bump ($\times 1.03$) on 1st/last days of month.

5. **Holiday & Festival Surges ($M_{\text{hol}}$):**
   - Directional long-weekend flow ($\times 1.25$ outbound, $\times 1.28$ inbound return).
   - Pre-holiday outbound surge ($\times 1.15$) and post-holiday return flow ($\times 1.12$).
   - Destination festival draw ($\times 1.15$) for major regional festivals (Onam, Pongal, Diwali).
   - Combined holiday multiplier is capped at $\le 1.65$ to prevent compounding explosion.

6. **Vacation Effects ($M_{\text{vac}}$):**
   - School vacations introduce weekend family travel increments ($\times 1.08$).
   - College vacations stimulate student corridor travel ($\times 1.04$ to $\times 1.05$).

### C. Stochastic Negative Binomial Sampling
To generate discrete passenger counts with realistic party-size overdispersion ($\text{Var} > \text{Mean}$), arrival volume is sampled from a Negative Binomial distribution:

$$Y_{ijst} \sim \text{NegBinomial}(n, p)$$

Where dispersion parameter $\alpha = 0.04$:
$$n = \frac{1}{\alpha} = 25.0, \quad p = \frac{n}{n + \mu_{ijst}} = \frac{25.0}{25.0 + \mu_{ijst}}$$

- **Rationale:** Models group arrivals (solo travelers, pairs, families of 3–4).
- **Properties:** Produces strictly non-negative integers natively without artificial truncation or negative rounding.
- **Reproducibility:** Seeded with `DEFAULT_RANDOM_SEED = 42`.

### D. Worked Demand Calculation Example
**Trip:** `TRIP_R_BLR_COK_20241031_DEP_04`
- **Route:** Bengaluru $\to$ Kochi (`R_BLR_COK`), $d = 540$ km ($d \ge 250$).
- **Departure:** 2024-10-31 (Thursday), 22:30 (`DEP_04`, Night Sleeper).
- **Date Features:** Diwali festival day (`is_holiday_ka = True`, `is_holiday_kl = True`, `is_month_end = True`, Season = Post-Monsoon).
1. $\text{BaseRate} = 37.5 \cdot 0.8868 \cdot 1.28 = 42.57$
2. $M_{\text{slot}} = 1.28$ (Long distance night sleeper)
3. $M_{\text{dow}} = 0.96$ (Thursday)
4. $M_{\text{season}} = 1.08 \cdot 1.03 = 1.112$ (Post-Monsoon + Month-end)
5. $M_{\text{hol}} = 1.15 \cdot 1.06 = 1.219$ (Destination & Origin Diwali holiday)
6. $M_{\text{vac}} = 1.00$
7. $\mu = 42.57 \cdot 1.28 \cdot 0.96 \cdot 1.112 \cdot 1.219 \approx 70.93$ passengers expected.
8. With $\mu = 70.93$, $p = 25 / (25 + 70.93) = 0.2606$.
9. Sampling yields $Y = 74$ passengers.
10. Operational outcomes: `bus_capacity = 45`, `capacity_exceeded = True`, `estimated_unserved_demand = 74 - 45 = 29` unserved passengers (simulating a sold-out run with 29 waitlisted travelers).

### E. Summary of Exploratory Data Analysis (EDA) Findings
- **Total Simulated Trips:** 87,680 scheduled departures.
- **Mean Passenger Demand:** **35.76 seats** (average load factor: **79.5%**).
- **Standard Deviation:** **15.70** (strong variation, IQR: 25.0 – 44.0).
- **Capacity-Exceeded Rate:** **22.07%** of trips exceed the 45-seat physical limit, generating **263,766 total unserved passengers** across the 3-year network. This figure is an emergent result of the specific parameters and multipliers chosen in this synthetic simulation scenario, not an empirically verified measurement of actual commercial bus operations.
- **Top Corridor:** Bengaluru $\leftrightarrow$ Chennai (`R_BLR_MAA`: mean 49.84).
- **Lowest Corridor:** Thrissur $\leftrightarrow$ Kozhikode (`R_TCR_CCJ`: mean 23.70).
- **Demand Range & Outliers:** Minimum demand is 2 passengers (off-peak midday run); maximum demand reaches 195 passengers on peak festive sleeper departures (representing intense booking interest rather than physical bus occupancy).
- **Artifacts:** Full analysis available in [reports/trip_demand_eda.md](../reports/trip_demand_eda.md) and [notebooks/01_trip_demand_eda.ipynb](../notebooks/01_trip_demand_eda.ipynb).

---

## 8. Reproducibility & Validation Principles

1. **Fixed Seed:** All pseudo-random sampling (NumPy RNG) is initialized with seed `42`.
2. **Boundary Realism:** Passenger demand is constrained by physical coach capacity ($N = 45$ seats) with latent unfulfilled demand tracking (simulating waitlists/sold-out runs).
3. **Audit Trail:** Every synthetic artifact records generation timestamp, configuration parameters, and schema checksums.

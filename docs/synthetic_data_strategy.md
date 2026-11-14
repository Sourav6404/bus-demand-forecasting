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

Each of the 8 cities is assigned standardized, empirical-anchored scores normalized between 0.0 and 10.0 across six core dimensions:

1. **Urban Scale ($S_u$):** Population density, urban agglomeration size, and metropolitan tier status.
2. **Employment Score ($S_{emp}$):** Concentration of IT, manufacturing, industrial corridors, and corporate headquarters.
3. **Education Score ($S_{edu}$):** Density of premier universities, engineering/medical colleges, and student migration catchment.
4. **Tourism Score ($S_{tour}$):** Heritage, religious, pilgrimage, eco-tourism, and leisure visitor attractions.
5. **Economic Activity Score ($S_{econ}$):** Commercial banking, wholesale trade, retail intensity, and per-capita GDP contribution.
6. **Migration Score ($S_{mig}$):** Net interstate and interdistrict workforce migration flux.

### Derived Indices
- **Demand Generation Index ($DGI$):**
  $$DGI = 0.30 \cdot S_u + 0.25 \cdot S_{emp} + 0.20 \cdot S_{econ} + 0.15 \cdot S_{mig} + 0.10 \cdot S_{edu}$$
- **Demand Attraction Index ($DAI$):**
  $$DAI = 0.25 \cdot S_{emp} + 0.25 \cdot S_{econ} + 0.25 \cdot S_{tour} + 0.15 \cdot S_{edu} + 0.10 \cdot S_u$$
- **City Mobility Index ($CMI$):**
  $$CMI = \frac{DGI + DAI}{2} \cdot \left(1 + 0.05 \cdot \text{scale\_bonus}\right)$$

---

## 4. Behavioral Demand Dynamics & Modifiers

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

## 5. Reproducibility & Validation Principles

1. **Fixed Seed:** All pseudo-random sampling (NumPy RNG) is initialized with seed `42`.
2. **Boundary Realism:** Passenger demand is constrained by physical coach capacity ($N = 45$ seats) with latent unfulfilled demand tracking (simulating waitlists/sold-out runs).
3. **Audit Trail:** Every synthetic artifact records generation timestamp, configuration parameters, and schema checksums.

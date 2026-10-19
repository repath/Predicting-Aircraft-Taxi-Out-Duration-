# Predicting Aircraft Taxi-Out Duration & Surface Congestion
**Student Identifier / Roll No:** `2606MGAIL130`  
**Coursework:** JNTU Hyderabad — Aviation Data Science and Operations Research  
**Data Source:** Bureau of Transportation Statistics (BTS) Airline On-Time Performance (2025)  

---

### ABSTRACT
Aircraft taxi-out duration, measured from gate pushback (the moment an aircraft begins its ground journey) to wheels-off (the precise timestamp at which the aircraft loses ground contact), serves as a fundamental operational efficiency indicator for modern air transportation systems. Long, unpredictable taxi-out durations cause significant excess jet fuel consumption, propagate delay cascades across the National Airspace System (NAS), and severely complicate airport runway departure sequencing.

This project develops an end-to-end, machine learning-driven decision support framework to forecast taxi-out duration prior to or at gate pushback, translating predictions into actionable operational guidance for Air Traffic Control (ATC) and airline dispatchers under Airport Collaborative Decision Making (A-CDM) principles. While flight delays in operational airspace depend on dozens of interconnected variables, our framework demonstrates that combining pre-departure scheduling waves with surface queue density provides high-precision estimates suitable for tactical ground operations.

---

### ALGORITHM ADOPTION & BENCHMARKING PLAN
We implement and evaluate three distinct model architectures across a strictly chronological train/test boundary (Months 1–10 for model training; Months 11–12 for out-of-time evaluation):

1. **Linear Regression (Baseline):** Provides an interpretable linear benchmark utilizing standardized continuous metrics and one-hot categorical encodings.
2. **Random Forest Regressor (Bagging Ensemble):** Reduces model variance by averaging 100 independently trained decision trees, effectively capturing non-linear interactions between gate delays and airport surface geometry.
3. **XGBoost Regressor (Gradient Boosting Ensemble):** Employs sequential gradient tree boosting to iteratively correct residual errors, excelling at capturing acute delay escalation thresholds when concurrent departures surge during bank waves.

#### Empirical Model Performance on Held-Out Test Set (12,001 Flights):

| Model | MAE (min) | RMSE (min) | $R^2$ Score | Within $\pm 3$ min | Within $\pm 5$ min |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression Baseline** | 4.17 | 10.04 | 0.0166 | 54.6% | 78.9% |
| **Random Forest Regressor** | 4.04 | 10.00 | 0.0239 | 57.5% | 82.6% |
| **XGBoost Regressor (Top Performer)** | **3.88** | **9.89** | **0.0452** | **58.8%** | **84.9%** |

Data understanding and exploratory analysis reveal that departure-bank timing, airfield geometry, and rolling 30/60-minute surface congestion provide significantly stronger predictive context than stage distance alone. The accompanying interactive dispatch simulator provides predicted taxi-out decomposition, sensitivity curves, operational risk tiers, and what-if pushback scenarios.

---

### OPERATIONAL SIGNIFICANCE & A-CDM INTEGRATION
The engineering implementation translates predictive models into practical airfield decision support:
1. **Pre-Departure Gate-Hold Advisories:** When surface queues exceed critical thresholds, holding aircraft at departure gates for 5–10 minutes with engines off saves approximately $130\text{ kg}$ of Jet-A fuel and prevents over $410\text{ kg}$ of $\text{CO}_2$ emissions per delayed flight.
2. **Departure Metering Program (DMP) Synchronization:** High-confidence Target Takeoff Times ($\text{TTOT} = \text{DEP\_TIME} + \text{TAXI\_OUT}$) allow tower coordinators to sequence runway queues smoothly, preventing apron gridlock.
3. **Decision Support Framing:** The interactive simulator provides what-if sensitivity analyses, uncertainty quantification, and operational risk tiers (Normal, Moderate, High, Severe) to augment rather than replace air traffic controller judgment.

---

### CONCLUSION & FUTURE EXTENSIONS
Empirical evaluation demonstrates that machine learning provides actionable, highly accurate taxi-out estimates (84.9% within FAA $\pm 5$-minute tolerance under XGBoost). The primary operational limitations reflect the exclusion of real-time METAR weather phenomena, active runway configuration changes, and arrival surface crossing bottlenecks. Future research will incorporate live arrival pressure and ASDE-X radar telemetry to enhance generalization across the wider National Airspace System.

**Keywords:** Taxi-Out Duration, Machine Learning, XGBoost, Random Forest, Surface Congestion, Airport Collaborative Decision Making (A-CDM), Departure Metering, BTS Aviation Data.

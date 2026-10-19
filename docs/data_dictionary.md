# Aviation Data Dictionary & Feature Specifications

**Project:** Aircraft Taxi-Out Duration Prediction Pipeline  
**Dataset:** BTS Marketing Carrier On-Time Performance (2025 Full Year)  
**Primary Hub Scope:** Dallas/Fort Worth International Airport (DFW) & Major US Hubs  
**Submission Identifier:** `2606MGAIL130`  

---

## 1. Raw Bureau of Transportation Statistics (BTS) Features

The raw dataset is extracted from the US Department of Transportation Bureau of Transportation Statistics (BTS) Airline On-Time Performance database (`data/raw/bts_carrier_ontime_sample.csv`).

| Column Name | Data Type | Physical Range / Values | Operational Definition & Usage |
| :--- | :--- | :--- | :--- |
| `FL_DATE` | Date (`YYYY-MM-DD`) | 2025-01-01 to 2025-12-31 | Flight departure date. Used to extract temporal trends and partition train/test sets. |
| `OP_UNIQUE_CARRIER` | Categorical (String) | AA, DL, UA, WN, B6, AS, NK, F9 | Operating marketing airline code. Encodes carrier pushback bank policies and terminal lease locations. |
| `ORIGIN` | Categorical (String) | DFW, ATL, ORD, DEN, LAX, CLT, JFK | Departure airport IATA code. Dictates airfield layout complexity, runway count, and taxi distances. |
| `DEST` | Categorical (String) | ORD, ATL, LAX, LGA, MIA, DEN, SFO, etc. | Destination airport IATA code. Dictates standard instrument departures (SIDs) and runway heading. |
| `CRS_DEP_TIME` | Integer (HHMM) | 0500 to 2355 (00:00–23:59) | Computerized Reservation System scheduled local departure time. Governs scheduled bank congestion. |
| `DEP_TIME` | Float / Int (HHMM) | 0000 to 2359 | Actual aircraft gate pushback time. Used for actual queue sequencing and estimated wheels-off time. |
| `DEP_DELAY` | Numeric (Minutes) | -25.0 to +180.0 min | Departure delay relative to schedule ($\text{DEP\_TIME} - \text{CRS\_DEP\_TIME}$). Represents initial ramp queue status. |
| `DISTANCE` | Numeric (Statute Miles) | 250 to 2,600 miles | Flight stage distance. Acts as a proxy for aircraft weight class and assignment to longer outer runways. |
| `CANCELLED` | Binary (0 / 1) | 0 = Operated, 1 = Cancelled | Flight cancellation indicator. Records with `CANCELLED == 1` are excluded during Step 2 preprocessing. |
| `DIVERTED` | Binary (0 / 1) | 0 = Normal, 1 = Diverted | Flight diversion indicator. Diverted flights are removed from nominal operational modeling. |
| `TAXI_OUT` | Numeric (Minutes) | 5.0 to 120.0 min | **Target Variable:** Elapsed time between gate pushback and takeoff roll ($\text{WHEELS\_OFF} - \text{DEP\_TIME}$). |
| `WHEELS_OFF` | Float / Int (HHMM) | 0000 to 2359 | Actual airborne takeoff time. Used to compute ground truth target duration. |

---

## 2. Engineered Features for Machine Learning

Engineered in `src/feature_engineering.py` and output to `data/processed/train_features.csv` and `data/processed/test_features.csv`.

| Feature Name | Type | Mathematical Formulation | Operational Significance |
| :--- | :--- | :--- | :--- |
| `HOUR` | Integer | $\lfloor\text{CRS\_DEP\_TIME} / 100\rfloor \pmod{24}$ | Local departure hour ($0 \dots 23$). |
| `DAY_OF_WEEK` | Categorical / Int | $0 = \text{Monday}, \dots, 6 = \text{Sunday}$ | Identifies weekly business travel surges (Mon/Thu) vs leisure peaks (Fri/Sun). |
| `MONTH` | Integer | $1 \dots 12$ | Captures seasonal weather trends (summer convective storms, winter de-icing). |
| `SIN_HOUR` | Continuous | $\sin\left(\frac{2\pi \cdot \text{HOUR}}{24}\right)$ | Cyclical diurnal time encoding preserving continuous 23:59 to 00:00 transition. |
| `COS_HOUR` | Continuous | $\cos\left(\frac{2\pi \cdot \text{HOUR}}{24}\right)$ | Orthogonal cyclical diurnal time encoding. |
| `CONGESTION_1H` | Continuous / Int | $\sum_{i \in \text{Origin}} \mathbf{1}_{[t-60\text{m}, t]}$ | Rolling count of concurrent departures from the same airport in the preceding 60 minutes. |
| `CONGESTION_30M` | Continuous / Int | $\sum_{i \in \text{Origin}} \mathbf{1}_{[t-30\text{m}, t]}$ | Rolling count of concurrent departures in the immediate preceding 30 minutes. |
| `ORIGIN_TAXI_MEAN` | Continuous | Smoothed Empirical Bayes Mean | Historical baseline taxi-out duration for the origin airport (computed strictly on Jan–Oct train set). |
| `CARRIER_TAXI_MEAN`| Continuous | Smoothed Empirical Bayes Mean | Historical baseline taxi-out duration for the carrier (computed strictly on Jan–Oct train set). |

---

## 3. Data Cleaning & Transformation Audit

| Step | Rule Applied | Justification | Action |
| :--- | :--- | :--- | :--- |
| **3.1 Cancellation Filter** | `CANCELLED == 1` | Flight never entered the taxiway. | Excluded from dataset. |
| **3.2 Diversion Filter** | `DIVERTED == 1` | Irregular operations. | Excluded from dataset. |
| **3.3 Invalid Taxi Filter** | `TAXI_OUT <= 0` or `NaN` | Physically impossible negative or zero elapsed times. | Excluded from dataset. |
| **3.4 Outlier Capping** | `TAXI_OUT > 120.0` | Represent gate return mechanical holds or runway closures rather than queue dynamics. | Capped/filtered at 120 minutes. |
| **3.5 Temporal Split** | $\text{MONTH} \le 10 \implies \text{Train}$; $\text{MONTH} > 10 \implies \text{Test}$ | Mimics real-world prospective deployment; zero future lookahead. | Preserved chronological boundary. |

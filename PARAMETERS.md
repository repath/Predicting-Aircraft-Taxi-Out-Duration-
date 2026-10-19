# Taxi-Out Dispatch AI Parameters

This document explains the user-facing parameters, internal feature names, model choices, presets, outputs, and important layout values used by the Streamlit application.

The application separates display labels from model values. Users see readable airline and airport names, while the prediction pipeline continues to receive the original airline and airport codes.

## 1. Prediction Model Engine

Location: `streamlit_app.py`, model selector near the top of the sidebar.

| Display option | Internal value | Model file | Description |
|---|---|---|---|
| Random Forest Regressor (Recommended) | `rf` | `models/random_forest_regressor.joblib` | Tree ensemble that captures non-linear relationships and interactions between delay, congestion, time, carrier, and airport features. |
| XGBoost Regressor | `xgb` | `models/xgboost_regressor.joblib` | Gradient-boosted tree model. It is useful when small changes in congestion or delay have non-linear effects on taxi-out time. |
| Linear Regression Baseline | `lr` | `models/linear_regression_baseline.joblib` | Simple baseline model. It is easier to interpret but does not capture complex non-linear relationships as well as the ensemble models. |

The selected model is passed to `predict_single_flight(flight_info, model_type=model_type)`.

## 2. Scenario Presets

Location: `streamlit_app.py`, `Load Scenario Preset` selector.

| Preset | Carrier | Departure | Arrival | Scheduled departure | Actual pushback | Delay | Distance | 1-hour congestion | 30-minute congestion |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Custom | AA | DFW | ORD | 08:30 | 08:35 | 5 min | 802 mi | 18 | 10 |
| DFW Morning Peak Push | AA | DFW | ORD | 08:30 | 08:35 | 5 min | 802 mi | 18 | 10 |
| DFW Midday Off-Peak | AA | DFW | MIA | 14:00 | 14:00 | 0 min | 1,121 mi | 5 | 2 |
| JFK Evening Congested Peak | B6 | JFK | LAX | 18:45 | 19:10 | 25 min | 2,475 mi | 22 | 12 |

`Custom` uses the same initial values as the default DFW morning scenario until the user changes them.

## 3. Flight Parameters

### 3.1 Carrier

The sidebar displays airline names. The model receives the corresponding carrier code in `OP_UNIQUE_CARRIER`.

| Display name | Internal code |
|---|---|
| American Airlines | `AA` |
| Delta Air Lines | `DL` |
| United Airlines | `UA` |
| Southwest Airlines | `WN` |
| JetBlue Airways | `B6` |
| Alaska Airlines | `AS` |
| Spirit Airlines | `NK` |
| Frontier Airlines | `F9` |

Carrier is a categorical feature. The preprocessor converts it into one-hot encoded columns before model prediction.

### 3.2 Departure and Arrival

The sidebar displays the airport name followed by its IATA code, for example `DALLAS/FORT WORTH - DFW`. The model receives only the code.

| Display label | Internal code | Role |
|---|---|---|
| DALLAS/FORT WORTH - DFW | `DFW` | Departure or arrival |
| ATLANTA - ATL | `ATL` | Departure or arrival |
| CHICAGO O'HARE - ORD | `ORD` | Departure or arrival |
| DENVER - DEN | `DEN` | Departure or arrival |
| LOS ANGELES - LAX | `LAX` | Departure or arrival |
| CHARLOTTE - CLT | `CLT` | Departure or arrival |
| NEW YORK (JFK) - JFK | `JFK` | Departure or arrival |
| NEW YORK (LAGUARDIA) - LGA | `LGA` | Departure or arrival |
| SAN FRANCISCO - SFO | `SFO` | Departure or arrival |
| PHOENIX - PHX | `PHX` | Departure or arrival |

`ORIGIN` is used by the model. `DEST` is retained in the flight information and displayed in the operational summary, but it is not part of the final model feature list in the current implementation.

### 3.3 Flight Date

- UI label: `Flight Date`
- Internal field: `FL_DATE`
- Default: `2025-11-15`
- Format passed to inference: `YYYY-MM-DD`

The date is converted into temporal features. The weekday becomes `DAY_OF_WEEK`; the month is also calculated for processing, although `MONTH` is not included in the final model feature list.

### 3.4 Scheduled Departure

- UI label: `Scheduled Departure (HHMM)`
- Internal field: `CRS_DEP_TIME`
- Range: `0` to `2359`
- Step: `5`
- Examples: `830` means 08:30; `1845` means 18:45.

The scheduled hour is extracted from this value. The hour is clamped to the valid range of 0-23 and is used to calculate cyclic time features.

### 3.5 Actual Departure / Pushback

- UI label: `Actual Departure / Pushback (HHMM)`
- Internal field: `DEP_TIME`
- Range: `0` to `2359`
- Step: `5`

This value is used to calculate the estimated wheels-off time after the predicted taxi-out duration. It is not included directly in the final model feature list.

### 3.6 Departure Delay

- UI label: `Departure Delay (minutes)`
- Internal field: `DEP_DELAY`
- Type: decimal number
- Step: `1.0` minute

This represents the departure delay already present before or at pushback. Larger delays can be associated with operational disruption and increased taxi-out time.

### 3.7 Stage Distance

- UI label: `Stage Distance (miles)`
- Internal field: `DISTANCE`
- Type: decimal number
- Step: `50.0` miles

This is the scheduled flight distance. It is used as a continuous model feature and can act as a proxy for route length and aircraft operating pattern.

## 4. Surface Traffic Density Parameters

### 4.1 Origin Pushbacks Past 1 Hour

- UI label: `Origin Pushbacks Past 1 Hr`
- Internal field: `CONGESTION_1H`
- Range: `1` to `40`
- Default: scenario-dependent

This is the number of recent pushbacks at the origin airport during the previous hour. Higher values represent a busier departure surface and generally increase the expected queueing component.

The dashboard classifies the value as:

| Value | Classification |
|---:|---|
| Less than 8 | Low (Free Flow) |
| 8 to 14 | Moderate (Standard Push) |
| 15 to 21 | High (Peak Bank) |
| 22 or more | Severe (Surface Queueing) |

### 4.2 Origin Pushbacks Past 30 Minutes

- UI label: `Origin Pushbacks Past 30 Min`
- Internal field: `CONGESTION_30M`
- Range: `0` to `25`
- Default: scenario-dependent

This is a shorter-term congestion signal. It helps distinguish a recent surge from a longer, steady traffic level.

## 5. Internal Model Feature Schema

The inference code assembles the following final feature list:

```text
DEP_DELAY
DISTANCE
CONGESTION_1H
CONGESTION_30M
SIN_HOUR
COS_HOUR
ORIGIN_TAXI_MEAN
CARRIER_TAXI_MEAN
OP_UNIQUE_CARRIER
ORIGIN
DAY_OF_WEEK
```

### Continuous features

| Feature | Meaning |
|---|---|
| `DEP_DELAY` | Departure delay in minutes. |
| `DISTANCE` | Flight stage distance in miles. |
| `CONGESTION_1H` | Pushbacks from the origin in the previous hour. |
| `CONGESTION_30M` | Pushbacks from the origin in the previous 30 minutes. |
| `SIN_HOUR` | Sine encoding of scheduled departure hour. |
| `COS_HOUR` | Cosine encoding of scheduled departure hour. |
| `ORIGIN_TAXI_MEAN` | Historical average taxi-out value for the origin airport. |
| `CARRIER_TAXI_MEAN` | Historical average taxi-out value for the carrier. |

The continuous columns are standardized with `StandardScaler`.

### Categorical features

| Feature | Meaning |
|---|---|
| `OP_UNIQUE_CARRIER` | Airline operating the flight, represented by its code. |
| `ORIGIN` | Departure airport code. |
| `DAY_OF_WEEK` | Numeric weekday derived from `FL_DATE`, where Monday is 0 and Sunday is 6. |

The categorical columns are one-hot encoded with unknown values ignored.

### Derived temporal features

`HOUR`, `MONTH`, and the cyclic hour encodings are derived from the selected flight date and scheduled departure time. `HOUR` and `MONTH` support preprocessing but are not directly included in the final feature list; `SIN_HOUR` and `COS_HOUR` are included.

### Target-encoding fallback

Historical encodings are loaded from `models/target_encoders.json`.

- Missing origin or carrier lookup: falls back to `global_mean`.
- If `global_mean` is unavailable: fallback value is `20.0` minutes.
- If `CONGESTION_1H` is missing outside the Streamlit UI: fallback is `8.0`.
- If `CONGESTION_30M` is missing: fallback is half of `CONGESTION_1H`.

## 6. Prediction Outputs

The prediction function returns:

| Output field | Explanation |
|---|---|
| `flight` | Carrier and route summary. |
| `predicted_taxi_out_minutes` | Predicted pushback-to-wheels-off duration, rounded to one decimal place. |
| `scheduled_dep` | Scheduled departure formatted as `HH:MM`. |
| `actual_dep_time` | Actual pushback formatted as `HH:MM`. |
| `estimated_wheels_off` | Actual pushback time plus predicted taxi-out duration. |
| `model_used` | Human-readable name of the selected model. |

## 7. Visual and Layout Parameters

These values control the appearance of the main cards and charts in `streamlit_app.py`.

### Header card

Location: `.main-header` CSS block.

| Parameter | Current value | Effect |
|---|---:|---|
| `padding` | `0.75rem 1.5rem` | Controls internal top/bottom and left/right spacing. Lower the first value to reduce height. |
| `border-radius` | `12px` | Rounds the header corners. |
| `margin-bottom` | `1.5rem` | Space below the header. |
| `h1 font-size` | `1.8rem` | Size of the title. |
| `h1 margin-bottom` | `0rem` | Distance between the title and subtitle. |
| `p font-size` | `0.9rem` | Size of the subtitle. |

### Metric cards

Location: `.metric-card`, `.metric-value`, and `.metric-sub` CSS blocks.

| Parameter | Current value | Effect |
|---|---:|---|
| `.metric-card padding` | `0.7rem` | Controls card height and internal spacing. |
| `.metric-value font-size` | `1.8rem` | Size of the main KPI value. |
| `.metric-value margin` | `0.2rem 0` | Vertical gap around the KPI value. |
| `.metric-sub font-size` | `0.8rem` | Size of the explanatory text under the KPI. |

### Chart heights

| Chart | Code variable | Height |
|---|---|---:|
| Pushback-to-airborne breakdown | `fig_bar` | `340` px |
| Congestion sensitivity curve | `fig_curve` | `340` px |
| Model performance comparison | `fig_comp` | `380` px |
| Feature importance chart | `fig_imp` | `380` px |
| Hub comparison chart | `fig_hubs` | `380` px |
| Departure-hour profile | `fig_hour` | `380` px |

Categorical x-axis labels use Plotly `<br>` line breaks where labels are long. Numeric x-axis charts do not require manual wrapping.

## 8. Important Editing Guidance

1. Keep display labels separate from internal codes. Change `carrier_names` or `airport_names` when changing what the user sees; do not change the model codes unless the training data and model artifacts are also rebuilt.
2. Keep the feature names in `predict.py` and `src/train_models.py` synchronized.
3. If a new carrier or airport is added to the selector, confirm that the trained encoders and target-encoding data support it.
4. If a model file is renamed, update `load_deployment_artifacts()` and the model selector together.
5. If a numeric range or default is changed, update the scenario preset documentation in this file as well.
6. The user-facing code changes are in `streamlit_app.py`; inference feature preparation is in `predict.py`; training feature definitions are in `src/train_models.py`.
## 9. Training Configuration

Location: `src/train_models.py`.

### Training and output paths

| Parameter | Current value | Purpose |
|---|---|---|
| `train_path` | `data/processed/train_features.csv` | Training feature dataset. |
| `test_path` | `data/processed/test_features.csv` | Out-of-time test feature dataset. |
| `models_dir` | `models` | Destination for serialized model pipelines and evaluation metadata. |
| `dashboard_data_dir` | `dashboard` | Destination for dashboard data generated during training. |
| `TARGET_COL` | `TAXI_OUT` | Regression target measured in minutes. |

The current split is chronological: months 1-10 are used for training and months 11-12 are used for testing. This is intended to reduce temporal leakage.

### Preprocessing

- Continuous features use `StandardScaler`.
- Categorical features use `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
- Unlisted columns are dropped by the `ColumnTransformer`.

### Model hyperparameters

| Model | Parameters |
|---|---|
| Linear Regression | `LinearRegression()` with default parameters. |
| Random Forest Regressor | `n_estimators=100`, `max_depth=15`, `min_samples_split=10`, `random_state=42`, `n_jobs=-1`. |
| XGBoost Regressor | `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `random_state=42`, `n_jobs=-1`. |

### Evaluation metrics

The training pipeline calculates:

- MAE: average absolute prediction error in minutes.
- RMSE: penalizes larger errors more heavily.
- R-squared: explained variance relative to the target baseline.
- Percentage within plus or minus 3 minutes.
- Percentage within plus or minus 5 minutes.
- Mean residual bias and residual standard deviation.
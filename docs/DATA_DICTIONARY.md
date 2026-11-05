# Data and feature dictionary

## Sources and target

`data/station_047_brentwood.csv` supplies the target observations. Other `data/station_*.csv` files supply candidate neighbor weather records. `data/station_coordinates.json` supplies the geographic metadata. The program uses the top-level measurement CSVs, not their legacy-schema copies.

All timestamps are converted to fixed PST (`Etc/GMT+8`) and aligned on an hourly grid. The same-hour neighbor observation is assumed available at the target forecast origin. There is no simulated sensor delivery delay.

## Raw local attributes

| CSV column | Meaning | Supplied units |
| --- | --- | --- |
| eto_in | Reference evapotranspiration | inches |
| precip_in | Precipitation | inches |
| sol_rad_ly_per_day | Solar radiation | supplied radiation field; hourly semantics not independently verified |
| vapor_pressure_mbar | Vapor pressure | mbar |
| air_temp_f | Air temperature | degrees F |
| relative_humidity_pct | Relative humidity | percent |
| dew_point_f | Dew point | degrees F |
| wind_speed_mph | Current wind speed | mph |
| wind_direction_deg | Wind direction | degrees |
| soil_temp_f | Soil temperature | degrees F |

Station ID, station name, date, day group and quality-control text are not direct predictors. The target is the class of `wind_speed_mph` one hour later, with fixed boundaries 3 and 7 mph.

## Extracted local attributes

| Feature | Definition |
| --- | --- |
| wind_lag_1, wind_lag_3 | Same-day wind one and three hours earlier |
| wind_mean_3 | Current and previous two same-day hours' mean, using available values |
| wind_std_3 | Sample SD over that trailing window, requiring at least two observed values |
| wind_change | Current wind minus one-hour lag |
| temperature_dewpoint_spread | Air temperature minus dew point |
| direction_sin, direction_cos | Wind direction transformed by sin/cos with a 360-degree period |
| hour_sin, hour_cos | Hour of day transformed with a 24-hour period |
| annual_sin, annual_cos | Day of year transformed with a 365.25-day period |

All temporal features reset at calendar-day boundaries. Partial or missing early-day histories are allowed and median-imputed inside the estimator's training fold.

## Spatial attributes

Prefix `n6_` denotes Davis, `n13_` Camino, and `n7_` Firebaugh/Telles. Each supplies current wind, air temperature, relative humidity and vapor pressure. The spatiotemporal arm also receives each station's `wind_lag_1` and `wind_mean_3`.

For wind, temperature and humidity, `neighbor_mean_...` is an inverse-distance weighted mean over observed neighbors. `local_minus_neighbor_...` is the target's current reading minus that weighted mean. Missing-neighbor means use only observed values and renormalized weights. No valid neighbor means a missing summary rather than a fabricated observation.

## Cleaning and audit rules

- Numeric parse failures and non-finite predictor values become missing.
- Nonnegative, observed future target wind is required.
- No target imputation, class resampling, artificial examples or label replacement is used.
- Median imputation and standard scaling fit only on the current training subset.
- Quality-control fields remain in the original data, but are not exhaustively filtered by this experiment.
- Neighbor missingness, coordinate weights, source hashes and exact feature names are recorded in `results/classification/metadata.json` and `neighbors.csv`.

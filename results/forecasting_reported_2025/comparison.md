# Brentwood 2025: user-reported results

Scores transcribed from the user's chat messages. No real-data experiment was run locally to obtain these scores. Prediction timestamps, checkpoints, configurations, neighbor exclusions, and exact split identity were not supplied for independent verification.

RMSE and MAE are in m/s; MAPE is percent. Bold marks the lowest reported error per season, horizon, and metric. Comparisons use supplied precision; spatial scores were supplied to fewer decimal places. All reported zero-target counts are zero, so MAPE equals MAPE_nonzero.

|Season|Horizon|RMSE L|RMSE L.1|RMSE Spatial L.1|MAE L|MAE L.1|MAE Spatial L.1|MAPE L|MAPE L.1|MAPE Spatial L.1|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|spring|1|0.010664|**0.010257**|0.010466|0.008682|**0.008202**|0.008485|0.567|**0.536**|0.575|
|spring|2|0.014505|0.014279|**0.013817**|0.012188|0.011989|**0.011546**|0.804|0.788|**0.767**|
|spring|3|0.027839|0.027216|**0.026147**|0.023496|0.022899|**0.021876**|1.552|1.509|**1.463**|
|summer|1|0.013240|**0.012260**|0.012280|0.009409|**0.008470**|0.008481|0.623|0.558|**0.557**|
|summer|2|**0.021914**|0.021960|0.022213|0.015080|**0.014995**|0.015168|**1.009**|1.031|1.035|
|summer|3|**0.038127**|0.038503|0.038894|0.026292|**0.025948**|0.026251|1.764|**1.762**|1.773|
|autumn|1|0.004493|**0.004272**|0.004635|0.003592|**0.003298**|0.003720|0.345|**0.318**|0.360|
|autumn|2|0.005055|0.005147|**0.004772**|0.004060|0.004113|**0.003753**|0.398|0.393|**0.369**|
|autumn|3|0.010045|**0.009210**|0.009267|0.008093|**0.007247**|0.007325|0.789|**0.697**|0.715|
|winter|1|0.008310|**0.007809**|0.008070|0.006639|**0.006188**|0.006315|0.461|**0.431**|0.437|
|winter|2|**0.009382**|0.010257|0.009760|**0.007204**|0.008178|0.007542|**0.510**|0.579|0.522|
|winter|3|0.016329|**0.012786**|0.013711|0.012702|**0.009911**|0.010545|0.893|**0.696**|0.743|

## Wins

|Metric|L|L.1|Spatial L.1|
|---|---:|---:|---:|
|RMSE|3|6|3|
|MAE|1|8|3|
|MAPE|2|6|4|
|Total|6|20|10|

L.1 wins the most reported metrics (20/36). Spatial L.1 wins 10 and L wins 6. Spatial L.1 wins all three metrics in spring horizons 2 and 3 and autumn horizon 2. This single reported run does not establish a reliable or operational forecasting advantage; the configured experiments retain full-season leakage.

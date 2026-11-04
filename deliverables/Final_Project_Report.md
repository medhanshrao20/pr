Next Hour Wind Regime Classification

Classical pattern recognition with temporal and circular feature extraction

Pattern Recognition semester project
Monsoon 2026
Submission deadline 5 November 2026

Abstract

This project classifies the next hour of wind speed at CIMIS Brentwood into three fixed categories using current weather observations. The study contains 16,806 usable instances, ten raw meteorological attributes and 22 attributes after feature extraction. Classical methods form the complete experimental comparison: logistic regression, shrinkage linear discriminant analysis and a random forest, with persistence as a practical baseline. All learning and model selection use 2024 observations. Five stratified folds keep calendar days together, and 2025 forms a separate future test set. The validation-selected random forest achieves 78.48% accuracy and 0.7894 macro-F1 on the 2025 test set. Its five-fold validation accuracy is 77.06 +/- 0.72 percentage points. Feature extraction raises logistic regression test macro-F1 from 0.7675 to 0.7758. The results support modest, measurable benefit from the extracted features on this station and period.

Problem and motivation

Wind varies over short intervals. Predicting a small number of next-hour regimes can support a simple monitoring or planning interface. This study evaluates a meteorological classification task. It does not estimate electricity output or validate a safety warning system. The category boundaries are project design choices rather than official meteorological warning thresholds.

Code repository

<link href="https://github.com/medhanshrao20/pr">https://github.com/medhanshrao20/pr</link>

Run Main.m from the repository root to regenerate the complete classification experiment and display its results. The supplied forecasting implementation remains separately archived without source edits.


---


1 Dataset and classification task

The supplied station_047_brentwood.csv contains 17,568 hourly records from 1 January 2024 to 2 January 2026 in fixed Pacific Standard Time. The experiment restricts observations to calendar years 2024 and 2025. CIMIS provides measured and calculated meteorological variables [1]. No additional data download is required.

| Quantity | Value |
| --- | --- |
| Raw records | 17,568 |
| Usable labelled instances | 16,806 |
| Training instances in 2024 | 8,413 |
| Held-out test instances in 2025 | 8,393 |
| Raw predictors | 10 |
| Predictors after feature extraction | 22 |
| Target classes | 3 |

Target definition

For a feature vector observed at hour t, the label is a category of wind_speed_mph at t+1. The next timestamp must be exactly one hour later and on the same calendar day. Missing or negative target wind values are excluded. This removes 762 rows from the raw file, including the excluded 2026 observations and invalid or cross-day target pairs. The model receives current wind as a valid predictor, but never receives next-hour wind as an input.

| Class | Next-hour speed | 2024 count | 2025 count |
| --- | --- | --- | --- |
| Low | Below 3 mph | 2,698 | 2,911 |
| Moderate | 3 to below 7 mph | 3,763 | 3,652 |
| High | 7 mph or higher | 1,952 | 1,830 |

Input attributes and units

The ten raw attributes are reference evapotranspiration (inches), precipitation (inches), solar radiation (the supplied sol_rad_ly_per_day field), vapor pressure (mbar), air temperature (degrees F), relative humidity (%), dew point (degrees F), current wind speed (mph), wind direction (degrees), and soil temperature (degrees F). Column names and units follow the provided CSV. No claim is made that the solar-radiation column label independently verifies its hourly measurement semantics.

Predictor missing values remain missing until median imputation inside each training pipeline. Non-finite predictor values become missing. Quality-control fields are retained in the archived CSV but are not used as predictors. A separate audit of every CIMIS quality flag is outside this experiment.


---


2 Feature extraction and classical models

The experimental contribution is an explicit feature-extraction comparison for a new multiclass task on the supplied weather data. The classifiers and transformations are established techniques. This work does not claim a new learning algorithm or reproduction of a published classification benchmark.

Temporal and circular features

Twelve added attributes describe information available at the observation hour: wind lags of one and three hours, a trailing three-hour wind mean and sample standard deviation, current wind minus its one-hour lag, temperature minus dew point, and sine/cosine pairs for wind direction, hour of day and annual position. Sine/cosine encoding represents the continuity between 359 and 0 degrees and between the end and beginning of a cycle. Each history calculation resets at a day boundary. No centered window, backward fill, full-season decomposition or future observation is used.

For angle theta, the circular representation is (sin(theta), cos(theta)). For direction, theta = pi times degrees / 180. For time, theta = 2 pi times hour / 24. Annual position uses 2 pi times (day of year minus 1) / 365.25. Partial trailing windows use only available observations. Missing lag values at the start of a day are imputed from the training median.

| Method | Configuration |
| --- | --- |
| Persistence | Predict the class of current wind speed. Impute missing current wind with training median. |
| Logistic raw | Ten raw predictors. Balanced class weights, L2 regularization, C=1, maximum 1,500 iterations. |
| Logistic features | Same classifier and settings with 22 predictors. This isolates the feature-set change. |
| LDA features | 22 predictors. Linear discriminant analysis, lsqr solver, automatic covariance shrinkage. |
| Random forest features | 22 predictors. 160 trees, maximum depth 14, minimum leaf size 3, balanced subsample weights, seed 42. |

Each learned model uses median imputation and standard scaling before classification. The forest does not require scaling, but uses the same pipeline for consistency. The scaler and imputer fit only on that fold's training partition. Neither PCA nor learned feature selection is used. Settings are fixed before the 2025 evaluation. The feature comparison is an ablation study rather than a claim of statistically significant superiority.

Deep learning

The course experiment uses classical models only. The supplied KANInformer paper and forecasting source are preserved as related work in separate folders. Their continuous forecasting errors are not compared numerically with classification scores, because the target and evaluation protocol differ.


---


3 Evaluation protocol and validation results

All 2024 rows form the development pool. StratifiedGroupKFold with five folds, shuffle enabled and seed 42 approximately preserves class proportions while assigning each calendar day entirely to one fold [2]. Every fold contains all three classes. A candidate trains on four folds and validates on the remaining fold. Day grouping prevents the same day's overlapping histories and next-hour targets from entering different folds. Adjacent days can remain correlated, so this validation estimates retrospective generalization within 2024. It is not a rolling forecast simulation.

The complete 2025 year is held out until the fixed candidate pipelines have been evaluated on 2024. The learned model with the highest mean validation macro-F1 is selected, then refitted on all 2024 data. Final test scores for all fixed candidates are retained for transparent comparison. Model choice follows validation, without changing thresholds, features or settings after inspecting test scores.

Metrics

Accuracy is the fraction of correctly classified instances. For each class, precision = TP / (TP + FP), recall = TP / (TP + FN), and F1 = 2PR / (P + R). Macro-F1 is the unweighted average of the three class F1 scores. Undefined ratios receive zero. Mean and sample standard deviation (ddof=1) summarize the five validation folds. These folds share training observations, so their spread is not a confidence interval or an independent repeated-run uncertainty estimate.

| Model | Accuracy mean +/- SD | Macro-F1 mean +/- SD |
| --- | --- | --- |
| Persistence | 74.49 +/- 0.40 | 75.15 +/- 0.45 |
| Logistic raw | 74.72 +/- 0.90 | 75.40 +/- 0.84 |
| Logistic features | 75.22 +/- 0.87 | 75.92 +/- 0.79 |
| LDA features | 74.20 +/- 0.82 | 75.03 +/- 0.90 |
| Random forest features | 77.06 +/- 0.72 | 77.62 +/- 0.80 |

Table 1. Five-fold 2024 validation scores expressed on a 0 to 100 scale. SD values are percentage points. The highest learned-model mean macro-F1 selects the random forest.

Feature extraction changes logistic validation macro-F1 by 0.52 percentage points. The random forest improves on persistence by 2.47 validation macro-F1 points. These are descriptive comparisons. No significance test was performed.

Audit artifacts

split_assignments.csv records every timestamp, class, year split and validation fold. metadata.json records class counts, fold day membership, versions, seed, feature names and the dataset SHA-256 hash. test_predictions.csv retains every model's final test predictions. check_results.py recomputes metrics and checks group isolation, the future holdout and preserved source hashes.


---


4 Held out 2025 results

| Model | Accuracy | Macro-F1 |
| --- | --- | --- |
| Persistence | 75.47% | 0.7607 |
| Logistic raw | 76.33% | 0.7675 |
| Logistic features | 77.12% | 0.7758 |
| LDA features | 75.21% | 0.7600 |
| Random forest features | 78.48% | 0.7894 |

Table 2. Scores from 8,393 held-out 2025 instances. The validation-selected random forest achieves 78.48% accuracy and 0.7894 macro-F1. Relative to persistence, this is an accuracy gain of 3.01 percentage points and a macro-F1 gain of 0.0288.

Selected model per class

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Low (<3 mph) | 0.8016 | 0.8327 | 0.8168 | 2911 |
| Moderate (3 to <7 mph) | 0.7616 | 0.7385 | 0.7499 | 3652 |
| High (>=7 mph) | 0.8020 | 0.8011 | 0.8015 | 1830 |

Figure 1. Confusion matrix for the validation-selected forest. Rows are true labels and columns are predictions. Most errors occur between adjacent wind categories. Only seven test cases confuse the lowest and highest categories directly.


---


5 Discussion and conclusions

Logistic regression with extracted features raises held-out accuracy from 76.33% to 77.12% and macro-F1 from 0.7675 to 0.7758. Since the classifier and training protocol are unchanged, this comparison measures the effect of the feature package for this experiment. It does not isolate which individual feature provides the gain.

The forest provides the strongest validation result among the learned candidates. Its held-out performance is also stronger than persistence in this dataset. Persistence is already competitive, reflecting the continuity of hourly wind. The moderate class remains difficult because observations near both category boundaries can be confused with neighboring classes. The per-class recall and confusion matrix expose this behavior more clearly than accuracy alone.

Limitations

This study uses one station and a single future year. The category thresholds are fixed engineering choices without domain validation. Day-group validation limits within-day overlap but cannot remove all weather dependence across days. The future-year test gives a separate temporal check, although it is not a second independent dataset. Missing predictors use median imputation, and no comprehensive quality-flag filtering was applied. Five-fold SD describes validation variation only. No deep-learning classification comparator, statistical significance claim, operational alarm claim or causal interpretation is included.

Conclusion

Classical multiclass classification is feasible on the supplied weather records with more than the required number of attributes and instances. A small temporal and circular feature set improves logistic regression over its raw-feature counterpart. The validation-selected forest provides 78.48% test accuracy and 0.7894 test macro-F1. The complete evaluation can be rerun through Main.m, and saved prediction-level artifacts support independent verification.

References

[1] California Department of Water Resources. California Irrigation Management Information System. Weather data and hourly report resources. https://cimis.water.ca.gov/ and https://cimis-proxy.water.ca.gov/Resources.aspx. Accessed 4 November 2026.

[2] scikit-learn developers. Cross-validation: evaluating estimator performance. Grouped and stratified cross-validation. https://scikit-learn.org/stable/modules/cross_validation.html. The executed environment uses scikit-learn 1.3.2.

[3] scikit-learn developers. Common pitfalls and recommended practices. Data leakage and preprocessing pipelines. https://scikit-learn.org/stable/common_pitfalls.html.

[4] Leng Z, Chen L, Yi B, Liu F, Xie T, Mei Z. Short-term wind speed forecasting based on a novel KANInformer model and improved dual decomposition. Energy 322, 135551, 2025. https://doi.org/10.1016/j.energy.2025.135551. Related forecasting work identified in the supplied spatial study.

The supplied spatial forecasting manuscript is retained as reference_paper/spatial_forecasting_paper.pdf. Its reported RMSE, MAE and MAPE are evidence for its forecasting task only, and are not the classification results reported here.


---


Appendix A Individual contributions

Team members must complete and verify this appendix before submission. Individual attribution cannot be inferred from source files or repository account names.

| Name and roll number | Actual work completed | Evidence or files |
| --- | --- | --- |
| To be supplied by the team | To be supplied by the team | To be supplied by the team |
| To be supplied by the team | To be supplied by the team | To be supplied by the team |

Record who performed problem formulation, dataset verification, feature design, classifier implementation, evaluation, report preparation and presentation work. Add or remove rows to match the actual team. Do not assign work to a member without verification. The editable CONTRIBUTIONS.md file provides a place to supply this information before revising this appendix.

Appendix B Reproducibility and source preservation

Repository: https://github.com/medhanshrao20/pr. The team must keep this repository accessible until grades are awarded. Main.m runs the new classification pipeline, prints all required metrics, saves the evidence and displays the confusion matrices. Python 3.10 or newer and MATLAB are prerequisites. If Python packages are missing, the launcher attempts installation from classification/requirements.txt using pip. Internet access is required only for that installation.

The recorded run used Python 3.11.9, NumPy 1.26.4, pandas 2.2.2, scikit-learn 1.3.2 and Matplotlib 3.10.0. The random seed is 42. A CPU is sufficient. Python-only execution is also supported with python -B classification/run_classification.py. Verification uses python -B classification/check_results.py. Re-running regenerates the classification result files in results/classification.

Original source files remain unchanged. reference_forecasting contains the copied forecasting program and data, and source_integrity.json records pre-existing source hashes. The new Main.m and classification folder implement an independent experiment. The original code's data cleaning, decomposition or neural model is not modified or silently reinterpreted as classification.

Dataset SHA-256

662a571563b0464da77fe588745277fda41cee26241fd87fcd6fdfea8720f9d7


---


Appendix C Per class test results for all models

| Model and class | Precision | Recall | F1 | N |
| --- | --- | --- | --- | --- |
| Logistic raw Low (<3 mph) | 0.7576 | 0.8770 | 0.8129 | 2911 |
| Logistic raw Moderate (3 to <7 mph) | 0.7863 | 0.6317 | 0.7006 | 3652 |
| Logistic raw High (>=7 mph) | 0.7401 | 0.8448 | 0.7890 | 1830 |
| Logistic features Low (<3 mph) | 0.7596 | 0.8825 | 0.8165 | 2911 |
| Logistic features Moderate (3 to <7 mph) | 0.7956 | 0.6438 | 0.7117 | 3652 |
| Logistic features High (>=7 mph) | 0.7554 | 0.8486 | 0.7993 | 1830 |
| LDA features Low (<3 mph) | 0.7477 | 0.7990 | 0.7725 | 2911 |
| LDA features Moderate (3 to <7 mph) | 0.7130 | 0.7232 | 0.7181 | 3652 |
| LDA features High (>=7 mph) | 0.8523 | 0.7350 | 0.7893 | 1830 |
| Random forest features Low (<3 mph) | 0.8016 | 0.8327 | 0.8168 | 2911 |
| Random forest features Moderate (3 to <7 mph) | 0.7616 | 0.7385 | 0.7499 | 3652 |
| Random forest features High (>=7 mph) | 0.8020 | 0.8011 | 0.8015 | 1830 |
| Persistence Low (<3 mph) | 0.7819 | 0.7870 | 0.7845 | 2911 |
| Persistence Moderate (3 to <7 mph) | 0.7204 | 0.7169 | 0.7186 | 3652 |
| Persistence High (>=7 mph) | 0.7791 | 0.7787 | 0.7789 | 1830 |

All results refer to the same held-out 2025 instances. Per-class metrics, confusion matrices and test predictions for every model are available as CSV files.
# Computed classification evidence

All tables in this directory come from `classification/run_experiment.py`. They describe a classification task rather than the retained forecasting RMSE/MAE/MAPE. The target station is Brentwood, labels are fixed next-hour Low / Medium / High categories, development uses 2024 and final evaluation uses 2025.

| Artifact | Contents |
| --- | --- |
| metadata.json | Dataset counts, input hashes, features, fixed neighbors, classifier settings, versions and fold counts |
| selection_before_test.json | Classifier choice using temporal validation macro-F1, written before final test scoring |
| split_assignments.csv | Every eligible target row, class, year split and validation fold |
| fold_scores.csv | Each feature condition/classifier/fold accuracy and macro-F1 |
| cv_summary.csv | Five-fold means and sample SD, ddof=1 |
| cv_predictions.csv | Out-of-fold predictions for every learned candidate and every 2024 development row |
| test_metrics.csv | Accuracy and macro-F1 for all fixed candidates on the same 2025 rows |
| per_class_test.csv | Precision, recall, F1 and support for every class and candidate |
| test_predictions.csv | Every final candidate's predicted and true class with origin/target timestamps |
| confusion_*.csv, confusion_*.png | Confusion counts and figures, with true classes in rows |
| paired_comparison.csv | Matched spatial-minus-temporal fold/test differences and paired day-bootstrap intervals |
| neighbors.csv | Geographic choice, distances, weights and missing-neighbor input audit |
| ablation_macro_f1.png | Held-out scores across four information conditions |
| paired_macro_f1.png | Paired macro-F1 effects and conditional bootstrap intervals |
| neighbor_layout.png | Target/neighbor coordinate plot without a basemap |

The scientific diagnostic figures are generated from saved evidence by `tools/build_figures.py`. They do not fit models or alter predictions.

The primary random-forest test macro-F1 is 0.789425 with temporal features and 0.789344 with spatiotemporal features. Its difference is approximately -0.000082. Logistic regression and LDA show small positive differences, but every paired macro-F1 interval includes zero. The tested spatial package does not establish consistent improvement. The full 2025 cohort had already been evaluated in earlier local-only work, so this follow-up is not a newly blind confirmatory test.

Recompute and verify the saved evidence with:

```text
python -B classification/check_results.py
```

Timing columns depend on machine load. Numeric classification results use fixed seeds; small cross-platform numerical differences remain possible. Fold SD is not a confidence interval. Day-bootstrap intervals are conditional on fitted models, assume day blocks suitable for resampling, and do not model all weather dependence across days.

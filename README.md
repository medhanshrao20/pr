# Next Hour Wind Regime Classification

Pattern Recognition semester project, Monsoon 2026. Deadline: 5 November 2026.

This repository contains a new classical multiclass experiment and the supplied spatial wind forecasting project, preserved separately without editing its source files. The course experiment predicts the next hour's wind category at CIMIS Brentwood.

## Run the complete program

Open MATLAB, navigate to this repository, and run:

```matlab
Main
```

`Main.m` runs the classification pipeline, prints accuracy, macro-F1, per-class precision/recall, cross-validation mean and sample standard deviation, and the selected model's confusion matrix. It saves all results and displays each model's confusion figure. No interactive choices or data downloads are needed. Run time depends on the CPU.

Prerequisites: MATLAB and Python 3.10+ available as `python` or `py -3`. The launcher installs missing Python packages from `classification/requirements.txt` using pip. That first-time installation needs internet access. The verified environment uses MATLAB R2025b, Python 3.11.9, NumPy 1.26.4, pandas 2.2.2, scikit-learn 1.3.2 and Matplotlib 3.10.0. Python dependencies are pinned to the executed versions.

Python-only execution and evidence verification:

```text
python -m pip install -r classification/requirements.txt
python -B classification/run_classification.py
python -B classification/check_results.py
```

## Deliverables

- [Project report PDF](deliverables/Final_Project_Report.pdf)
- [Editable report text and tables](deliverables/Final_Project_Report.md)
- [PowerPoint presentation](deliverables/Final_Presentation.pptx)
- [MATLAB entry point](Main.m)
- [Classical classification implementation](classification/run_classification.py)
- [Computed results and prediction-level evidence](results/classification)
- [Team contributions to complete](CONTRIBUTIONS.md)

**Before submission:** provide the actual team members, roll numbers and individual contributions. Appendix A of the report and the contribution slide currently identify that missing information. Verify the topic is unique within the course. Keep this repository accessible until grades are awarded.

## Data and protocol

The supplied Brentwood CSV has 17,568 records. The new experiment produces **16,806 usable instances**, with **10 raw meteorological attributes**, **22 attributes after feature extraction**, and **3 classes**. Labels use next-hour wind speed in mph: low `<3`, moderate `[3,7)`, and high `>=7`. These fixed project thresholds are not an official warning scale.

2024 provides 8,413 training instances. Five stratified validation folds group observations by calendar day. 2025 provides 8,393 final test instances. Imputation and scaling fit on training data only. Histories and target pairs stay inside their calendar day. Validation selects the learned model by mean macro-F1 before evaluating the future-year test set. Raw and extracted-feature logistic regression isolate the effect of feature extraction. Shrinkage LDA and a random forest provide further classical comparisons. Persistence predicts the current wind category as a next-hour baseline.

The course experiment does not use deep learning. The archived KANInformer forecasting paper has a different target and protocol, so its RMSE/MAE/MAPE are not presented as comparable classification results.

## Executed results

| Model | 2024 CV accuracy, mean +/- SD (%) | 2024 CV macro-F1, mean +/- SD | 2025 accuracy (%) | 2025 macro-F1 |
| --- | --- | --- | --- | --- |
| Persistence | 74.49 +/- 0.40 | 0.7515 +/- 0.0045 | 75.47 | 0.7607 |
| Logistic raw | 74.72 +/- 0.90 | 0.7540 +/- 0.0084 | 76.33 | 0.7675 |
| Logistic features | 75.22 +/- 0.87 | 0.7592 +/- 0.0079 | 77.12 | 0.7758 |
| LDA features | 74.20 +/- 0.82 | 0.7503 +/- 0.0090 | 75.21 | 0.7600 |
| Random forest features | 77.06 +/- 0.72 | 0.7762 +/- 0.0080 | 78.48 | 0.7894 |

SD is the sample standard deviation across five validation folds, not a confidence interval. Validation selects the random forest. All final test scores use the same 8,393 instances. Results were computed from the supplied data, including an end-to-end run through MATLAB `Main.m`.

The selected model's test confusion matrix has true classes as rows and predictions as columns:

| True / predicted | Low | Moderate | High |
| --- | --- | --- | --- |
| Low | 2424 | 484 | 3 |
| Moderate | 596 | 2697 | 359 |
| High | 4 | 360 | 1466 |

## Evidence and preservation

`results/classification/` includes fold scores, mean/SD, final test metrics, all per-class results, confusion tables/figures, per-instance test predictions, timestamp-level fold assignments and metadata with versions, class counts and the dataset SHA-256 hash. `classification/check_results.py` recomputes saved metrics and checks splits and preserved source hashes.

`reference_forecasting/` contains the supplied code and weather data. `reference_paper/spatial_forecasting_paper.pdf` contains the supplied manuscript. `source_integrity.json` fingerprints the pre-existing local source files. All 21 copied Python source files with counterparts in the supplied comparison root match byte for byte. The original project's five existing tests also passed. Those original tests use synthetic forecasting fixtures and do not establish forecasting performance on actual data.

The supplied forecasting metrics are preserved separately under `results/forecasting_reported_2025`. Their provenance identifies them as previously reported aggregate results, without prediction-level verification. They are not newly generated classifications.

## Limitations and attribution

One station and one future test year limit generalization. Adjacent-day weather dependence can remain in grouped validation. Class boundaries need domain validation for deployment. Predictor quality flags were not exhaustively filtered. The feature package improves the logistic baseline descriptively, without a statistical significance claim or new-algorithm claim.

Weather source: [California CIMIS](https://cimis.water.ca.gov/), [hourly report resources](https://cimis-proxy.water.ca.gov/Resources.aspx). Evaluation follows [scikit-learn grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) and [training-only preprocessing guidance](https://scikit-learn.org/stable/common_pitfalls.html). The supplied manuscript cites Leng et al., *Energy* 322, 135551 (2025), [DOI](https://doi.org/10.1016/j.energy.2025.135551), as related forecasting work. No blanket license is asserted over supplied third-party material.

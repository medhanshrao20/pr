# Verification record

Verified on 5 November 2026 for the new temporal versus spatiotemporal classification study.

- Ran `python -B classification/run_experiment.py` on the supplied actual station data. All four feature conditions, three learned classifiers and persistence completed.
- Ran the complete program again through `Main.m` in MATLAB R2025b. It regenerated the comparison, printed the metrics and displayed the figures without interactive choices.
- Ran `python -B classification/check_results.py` against the regenerated artifacts. All checks passed: saved accuracy and macro-F1 recompute, confusion counts recompute, test cohorts match across candidates, out-of-fold rows match their assigned groups, every validation row has exactly one out-of-fold prediction, fold means and sample SD recompute, temporal boundaries remain isolated, causal history features ignore later perturbations, and the original file hashes match.
- Confirmed all 50 source/data files present at baseline commit `b30b5d8660dd9834cee125cb8e989823c8a4e115` remain unchanged. Their hashes are in `original_source_manifest.json`.
- Confirmed both new independent classification entry points load and expose their command-line options.
- Generated scientific figures only from the saved prediction/metric evidence. The report builder does not fit or tune classifiers.
- Rendered and visually inspected all 15 PDF pages. Checked tables, figures, methodology flow, captions and page boundaries.
- Checked local Markdown links in the README, methodology, data dictionary and editable report.
- Confirmed there are no PowerPoint files in the repository.

The primary random forest has temporal test accuracy 0.7848206839 and spatiotemporal test accuracy 0.7839866555. Temporal macro-F1 is 0.7894252743 and spatiotemporal macro-F1 is 0.7893437542, subject to the full precision stored in the CSV. The small negative primary difference is retained. Positive secondary differences are also retained. No labels, scores or predictions were altered to support the research hypothesis.

This record does not establish a blind confirmatory result, cross-station generalization, deployment performance, sensor delivery latency, independent-dataset uncertainty, or quality-flag completeness. The 2025 cohort had previously been evaluated in a local-only study. The comparison uses a fixed follow-up protocol with its limitations stated in the report.

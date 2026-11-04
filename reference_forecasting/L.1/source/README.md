# Full-season VMD selection variant: original_dataleak.v1

This folder is based on the final sibling `original_dataleak` folder in this
repository, not the older implementation under `kaninformer_original`.
The code was prepared without executing VMD, EWT, checks, ridge fitting, or neural
training. It contains no new selected K/alpha values, results, or checkpoints.

## Agreed selection procedure

For each season, use the same cleaned full-season WS series as the baseline:

1. Try each VMD alpha in `{25, 100, 500, 2000}`.
2. For that alpha, run VMD for **every K from 2 through 12**, inclusive.
3. Reconstruct WS as the sum of the VMD modes and compute

   `RER = mean(abs(WS - sum(modes)) / maximum(abs(WS), 1e-12))`.

4. Select the K with the **lowest RER for that alpha**. There is no 3% stopping
   rule, no first-below-threshold selection, and no stabilization condition.
   Exactly equal RER values prefer the smaller K.
5. Using that `(alpha, selected K)` pair, construct the baseline CA/EWT expanded
   feature bank and training-fitted scaler. Fit a ridge probe with penalty 0.1 on
   flattened training windows and raw wind-speed targets. Compute validation
   RMSE across all three horizons in m/s. Test targets are not scored or passed
   to this ridge fit.
6. Compare these four validation RMSE scores. Select the alpha and its associated
   reconstruction-selected K with the lowest score. Exactly equal validation
   RMSE scores prefer the smaller alpha.
7. Rebuild the winning representation, verify that its VMD RER agrees with the
   selection report, and then use the existing ridge + KANInformer training flow.

This is **hybrid selection**: reconstruction selects K, prediction selects alpha.
It does not choose the lowest RER across all alphas, and it does not evaluate
prediction RMSE for every K. There are 44 VMD search candidates and four ridge
validation probes per season at default settings. Building each winning feature
bank and the final bank also reruns VMD, using the same deterministic settings.

Selecting K=12 is allowed. It means the lowest RER within the tested range, not
a globally optimal K or guaranteed best forecasting performance. The paper's
published K-selection rule is different; this is a documented local variant.

## Preserved baseline pipeline

- Same supplied CSV, unit conversions, seasonal intervals, and incomplete coverage.
- Full-season three-sigma cleaning, not-a-knot spline interpolation, and decomposition.
- Paper-listed weather inputs, with locally computed PCC saved separately.
- Same sample entropy estimator, SE > 0.4 aggregation, and eight-component EWT.
- All VMD modes + EWT components + selected weather + raw WS as input features.
- Training-fitted StandardScaler, seven observed hours, chronological 8:1:1
  partitions, and target-boundary exclusion.
- Ridge base predictor with penalty 0.1 and separately scaled neural residuals.
- Identical `model.py`: two encoder blocks, one decoder block, dimension 64,
  eight heads, spline KAN, and three simultaneous forecast horizons.
- Same Adam optimizer, cosine schedule, gradient clipping, seed 42, batch size 32,
  initial learning rate 0.001, 300-epoch limit, and early-stopping patience 30.
- Same combined raw-unit validation checkpoint selection and forecast metrics.

The input feature count changes automatically when K changes. The neural design
is preserved, while the input projection and ridge input width follow that count.
Full-season processing still permits future observations to influence features;
training-only scaling does not remove this retrospective dependence.

## Commands for later use only

**No commands below were executed while preparing this folder.**

From this folder, after installing the unchanged requirements, the usual full-run
entry point is:

```powershell
python run.py
```

It first executes the inherited method checks and the new mocked selection checks,
then a reduced smoke run, and then full training. The smoke run uses its reduced
240-row dataset/model but still applies the configured K/alpha selection rule.
Smoke outputs are removed by the inherited entry-point cleanup. The smoke run
does not provide final seasonal parameter values.

For an explicitly chosen fixed alpha, K is still selected by reconstruction:

```powershell
python run.py --no-tune-alpha --vmd-alpha 100
```

The default range can be changed explicitly with `--vmd-k-min` / `--vmd-k-max`.
Those bounds are saved in the configuration. `--require-paper-data` rejects
incomplete seasonal coverage; `--skip-smoke` bypasses the entry-point checks and
smoke training. The optional author architecture retains its existing restrictions:
use `--base-boost none --feature-bank caewt` with that mode. The default remains
the paper-oriented backbone with ridge residual prediction.

The existing baseline's GPU validation path is inherited unchanged; runtime
correctness and GPU execution have not been verified in this preparation-only task.

## Outputs when explicitly run later

All outputs belong to this folder's own `results/` directory by default:

- `<season>_k_rer_search.csv`: every `(alpha, K)` reconstruction score, fractions
  and percentages, status, per-alpha winner, and final selected-pair flag.
- `<season>_alpha_validation_search.csv`: the reconstruction-selected K for each
  alpha, its RER, ridge validation RMSE, status, and selected-pair flag.
- `<season>_vmd_selection.json`: complete successful selection report, candidates,
  tie rules, selected K/alpha/RER, and validation RMSE. Written before neural training.
- `<season>_predictions.csv`, `<season>_training.csv`, and checkpoints: existing
  baseline forecasting artifacts produced only when training is actually run.
- `summary.json`: per-season selected K/alpha, all selection records, decomposition
  diagnostics, actual settings, raw-data hash, and measured forecasting metrics.
- `metrics.csv` and `paper_format_results.md`: measured metrics and the original
  paper's separately labeled Table 9 reference values.

A failed/nonfinite candidate aborts selection; it is never silently skipped.
Search CSVs are retained as partial diagnostics on failure, with failed-row status.
Only `search_complete: true` in a successful selection JSON represents completion.
No fallback to the paper's K values is performed. The new folder contains no copied
baseline result files. Use a fresh `--output` directory for each future invocation
if previous outputs need to be retained. The root `results.py` still targets the
original two experiment folders; it does not automatically compare this variant.

## File-level changes

- `run.py`: new two-stage selector, configurable K bounds, selection diagnostics,
  selected-pair propagation, and new pipeline label. The neural training function
  and downstream feature/scaling/window logic were preserved.
- `decomposition.py`: shared finite/shape-checked RER helper. Both the selection
  sweep and the final decomposition report use the baseline's same RER formula.
- `checks.py`: added mocked checks for evaluating every K, selecting alpha by
  validation RMSE, deterministic ties, failure handling, and fixed-alpha searches.
  Checks are included for later execution but were not run.
- `README.md`, `REPRODUCTION_AUDIT.md`, and `.gitignore`: variant-specific
  instructions, provenance, limitations, and generated-output exclusions.
- `model.py`, `requirements.txt`, raw CSV, and retained author source: copied
  unchanged. No dependencies were added.

The original experiment folder, manuscript, saved predictions, and root comparison
script are outside this edit and remain unchanged.

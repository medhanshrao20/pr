# Brentwood-only comparison: L, L.1 and Spatial L.1

This new project compares **exactly three experiments**, not Spatial L:

|Runner|Selection|Forecast target and context|
|---|---|---|
|`L/run.py`|Fixed seasonal K; alpha selected by Ridge validation RMSE|Brentwood only|
|`L.1/run.py`|Minimum-RER K per alpha; winning alpha by Ridge validation RMSE|Brentwood only|
|`spatial_L.1/run.py`|Same Brentwood L.1 selection; neighbors use their own L.1 selection|Brentwood forecast with graph context from other stations|

## Protected original code

**No existing L, L.1, or spatial-stage code is edited.** `L/source/` and `L.1/source/` contain byte-for-byte copied Python source, original requirements, documentation and reference files. Old results/checkpoints/logs are not copied into the new experiment. The old dataset is replaced by a schema-converted view of the new Brentwood CSV in each copied `source/data/raw/hourly.csv`; the source Python files themselves are unchanged.

Run the wrapper `L/run.py` or `L.1/run.py`, **not `source/run.py`**. The latter intentionally retains its original 2020–2021 date constants. The separate wrapper selects actual new-data calendar seasons and calls the original cleaning, parameter-selection, feature/window construction and `train()` functions. It does not rewrite those functions or override any baseline model, Ridge or metric function. Original source hashes are written in run protocols. See `source_manifest.json` and `INTEGRITY.md` for source protection checks.

The only new architecture is in `common/graph_model.py`, applied **only** to the new Spatial L.1 experiment. That experiment temporarily supplies its own model and a Brentwood-only Ridge input adapter to a separately loaded copy of the L.1 training module; the copied source files remain unchanged. Nothing is injected into baseline L/L.1 runs.

## Data

The project includes the same **new 2024-onward Brentwood and other station CSVs used by the spatial project**, not the original 2020–2021 Brentwood dataset. All 14 raw station files and the coordinate asset are copied into this project's `data/`; runtime does not depend on any outside folder. Station 47 is Brentwood; the other 13 are potential context nodes, not forecast targets.

`prepare_data.py` creates legacy-schema CSV views without imputing, scaling, filtering or changing measurements. It converts `timestamp_pst` to the original Date/Hour schema, preserving fixed PST and 2400 meaning previous-date midnight. Source mph/Fahrenheit measurements are retained; **the original loader performs its original unit conversions**. Conversion audits and hashes are saved. Neighbor loading uses the same numeric mapping/conversions without falsely relabelling other stations as Brentwood.

Default year is the latest complete shared requested seasonal cycle (2025 for the supplied files). `--year 2025` sets it explicitly. Winter 2025 means December 2024 through February 2025; spring March–May 2025; summer June–August; autumn September–November. Separate seasonal models retain the original 80/10/10 chronological splits, not stage 4's whole-record 70/15/15 split. There is no concatenation of disconnected seasonal years.

Each selected station must structurally cover the requested complete periods. Missing internal measurements remain missing until handled by the **original** full-season 3-sigma cleaning and not-a-knot cubic spline imputation. No extra data normalization/QC policy is substituted. All baselines and Spatial L.1 use the same resulting Brentwood targets, origins and scored timestamps. Neighbor missingness does not drop additional Brentwood windows. The new comparison is therefore consistent with the requested retrospective leakage protocol, not an operational evaluation.

## What is preserved

Copied Config defaults retain: paper-mode KANInformer; window 7; three direct horizons; label length 4; d_model=64; heads=8; two encoder blocks and one decoder; factor 5; dropout 0; KAN hidden 20/grid 5; Adam lr=0.001; cosine schedule; max epochs 300; patience 30; batch 32; seed 42; no batch shuffling; training-fitted input StandardScaler; local Ridge regularization 0.1; residual **MinMaxScaler**; best combined validation MSE checkpoint; same feature bank (all VMD modes + EWT + selected meteorology + raw WS).

Published seasonal meteorological lists remain: spring/autumn WS+ET, summer WS+ET+AT+RH, winter WS. Original VMD/EWT configuration and full-season leakage remain. Optional explicit training overrides are recorded; use the same overrides for all three experiments. Old winning alpha/K values are not reused as optima for the new data.

* L fixes K=(11,10,11,10) in spring/summer/autumn/winter order; alpha candidates 25,100,500,2000 compete on Ridge validation RMSE.
* L.1 evaluates K=2..12 for every alpha. R(a,k)=mean(|y−sum modes|/max(|y|,1e−12)); choose k*(a)=argmin_k R(a,k), then a*=argmin_a J_validation(a,k*(a)). Every K is evaluated; first-below-3% is not used. The original implementation's exact tie rules remain.
* Spatial L.1 reuses the same-run Brentwood L.1 selected pair when that baseline output exists and its input/configuration matches. Otherwise it executes the same unchanged selection procedure. Neighbors are independently processed through L.1; their validation probes select neighbor decomposition parameters, but the final neural loss and output are Brentwood-only.

## Spatial L.1 architecture and mathematics

Each node has its own original feature bank and training-fitted scaler, so K/feature widths may differ. Packed tensors concatenate station features without padding; original Brentwood features occupy the leading F0 slots. The graph adapter projects each neighbor's features into F0 dimensions; Brentwood's representation remains its original feature vector before graph enrichment. Component index equality is not asserted to represent identical physical frequencies.

Default graph is geographic, symmetrized three-nearest-neighbor support, distance weight exp(−(d/sigma)^2), self loops, then row normalization. sigma is median supported distance; haversine distances use radius 6371.0088 km. Optional statistical/combined graphs use training-row correlations. Inherited full-season cleaning still affects those rows. Saving the complete graph shows which of the 13 other stations actually connects to Brentwood; default kNN does not guarantee all 13 directly contribute.

For projected historical node features p_j and Brentwood row A_0:

c(t)=sum_j A_0j p_j(t),

x'_0(t)=LayerNorm(x_0(t)+GELU(W_g c(t)+b_g)).

The unchanged copied KANInformer backbone processes x'_0 and predicts only Brentwood residuals. Ridge still fits **only the original Brentwood feature windows**, not the packed neighbor inputs:

g(x_0)=argmin_(W,b) sum_train ||Y_0−W vec(x_0)−b||²+0.1||W||².

The unchanged original train routine MinMax-scales the training residual Y_0−g(x_0), learns the neural correction, inverse-transforms it, and returns Yhat_0=g(x_0)+neural_correction. No station-wise auxiliary neural forecast loss is added. Only the graph/projection/normalization modules are new. Temporal backbone is constructed first to preserve the baseline seed's initial backbone weights before initializing added graph parameters; different subsequent graph computations can still affect stochastic optimization.

## CPU/GPU: preserving the source takes precedence

All comparison runners default to **CPU**. The protected local baseline `train()` constructs its validation input on CPU and calls the model without transferring that tensor to CUDA. Editing or monkey-patching L/L.1 to fix it would violate the instruction to leave their code untouched; this project does neither. Therefore baseline wrappers explicitly reject a CUDA selection before training rather than starting a run that would fail at validation. Run the three-way comparison with the default `--device cpu`.

Spatial L.1's newly written model adapter can accept CUDA (`--device cuda`) and moves its own forecast input correctly, but CPU baselines/GPU spatial training have different numerical execution conditions. GPU execution is not verified here. If GPU support in L/L.1 is later desired, it requires separately authorized changes or a separately authorized compatibility layer; no such change is made now.

## Exact-zero MAPE and scoring

Original `metrics()` rejects a test horizon containing zero wind speed. Its `train()` saves checkpoint, training history and predictions **before** that evaluation. The wrapper catches only that exact evaluation error after confirming predictions exist; it does not modify or replace the original metric function. It reports strict MAPE as undefined (NA) for affected horizons while retaining valid RMSE/MAE.

A separately labelled external diagnostic `mape_nonzero` excludes **exact zero targets only**, with eligible counts. It is not the stage 4 threshold-based MAPE (|y|>=0.1), and is not passed off as the original equation. All three models use the same targets and exclusion rule. Raw pre-cleaning targets and cleaning-change flags are saved alongside the original cleaned scoring targets. Imputed/outlier-replaced target accuracy should not be interpreted as accuracy against untouched observed measurements.

## Commands

Reuse the spatial virtual environment or install from this folder's copied requirements:

```powershell
cd "D:\Drive D\projects all mixed\journal\spatial\compare"
python -m pip install -r requirements.txt
python run_all.py --run-id brentwood_2025_seed42 --year 2025
```

This starts three experiments in sequence when **you** run it. It does not train Spatial L. Separate commands, using the same run ID and settings:

```powershell
python .\L\run.py --run-id brentwood_2025_seed42 --year 2025
python .\L.1\run.py --run-id brentwood_2025_seed42 --year 2025
python .\spatial_L.1\run.py --run-id brentwood_2025_seed42 --year 2025
python summarize.py --run-id brentwood_2025_seed42
```

Change run ID for each new run; outputs are not overwritten. Explicit override example: `python run_all.py --run-id seed43 --year 2025 --seed 43 --patience 30`. To restrict context, supply `--neighbor-stations 2 5 6 7` and a suitable `--neighbors`; this does not remove Brentwood. Full neighbor L.1 selection on all seasons can be expensive on CPU.

## Final result files

Each experiment writes its own `results/<run-id>/` containing source-compatible seasonal predictions, training histories, checkpoints, selected parameters, cleaned inputs, configs and provenance. The spatial run additionally saves neighbor selections, scalers, graphs and feature schema.

After all three finish, `results/<run-id>/comparison.md` at this comparison root is the **final table to inspect**. It bolds minimum errors and reports spatial improvement versus both L and L.1:

improvement(%)=100×(baseline_error−spatial_error)/baseline_error.

Positive means lower spatial error; undefined/zero-denominator comparisons are NA. Other outputs: `brentwood_comparison.csv`, `brentwood_metrics_long.csv`, `selected_parameters.csv`, `spatial_improvements.csv`, `best_results.csv`, `win_counts.csv` and `comparison_audit.json`. The collector verifies identical station, timestamps, targets, data hash, year and common configuration before producing a table. Exact numerical ties are retained, rather than inventing a winner from rounded values.

The old paper/old-period error table is not an apples-to-apples baseline and is not mixed into the new comparison. Lower errors in a single seed do not establish reliable spatial improvement. The original retrospective leakage protocol is retained and explicitly labelled.

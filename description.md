# Main Python files: Brentwood model comparison

This package contains copied Python files from the `compare` project. Their relative paths are preserved. Source files were copied without modification. The original root `description.md` and all existing project files were left untouched.

## Contents

|File or directory|Purpose|
|---|---|
|`run_all.py`|Runs L, L.1 and Spatial L.1 sequentially, then produces the comparison.|
|`prepare_data.py`|Converts new station CSVs into the original loader's CSV schema without training or changing measurement values.|
|`summarize.py`|Checks matching Brentwood targets/timestamps and configurations, then creates comparison tables, winners and percentage improvements.|
|`L/run.py`|New-data entry point for the protected prediction-based L baseline.|
|`L.1/run.py`|New-data entry point for the protected hybrid-selection L.1 baseline.|
|`spatial_L.1/run.py`|Entry point for Brentwood-only forecasting with neighboring-station graph context.|
|`common/__init__.py`|Marks the local shared module directory.|
|`common/experiment.py`|Handles copied-source loading, new seasonal dates, aligned data/windows, experiment configuration, training invocation and output audits.|
|`common/graph_model.py`|Defines the spatial target-only graph adapter and keeps Ridge restricted to Brentwood features.|
|`common/spatial_graph_utilities.py`|Supplies geographic distances, graph construction and related shared utilities.|
|`L/source/run.py` and `L.1/source/run.py`|Unchanged original implementations containing cleaning, feature/window construction, VMD parameter selection and Ridge–KANInformer training.|
|`L/source/model.py` and `L.1/source/model.py`|Unchanged original KANInformer temporal architecture.|
|`L/source/decomposition.py` and `L.1/source/decomposition.py`|Original VMD, sample-entropy and CA–EWT functions.|
|`L/source/checks.py` and `L.1/source/checks.py`|Original method checks retained with the copied source code.|

There are **18 Python files**, plus this description. Reference-only author code, the comparison project's separate test suite, caches, datasets, trained weights, results, requirements files and other documentation are intentionally excluded.

## Experiment meanings

* **L:** Brentwood-only Ridge + KANInformer residual correction. Fixed seasonal K values 11/10/11/10; alpha chosen from 25/100/500/2000 by Ridge validation prediction RMSE.
* **L.1:** Same prediction architecture. For each alpha, K=2..12 is selected by minimum reconstruction error; alpha is then selected by Ridge validation prediction RMSE.
* **Spatial L.1:** Adds neighboring-station graph information to the neural residual path. The final forecast and neural loss remain Brentwood-only, and Ridge still uses Brentwood's own features.

Full-season cleaning/decomposition leakage is retained. The comparison's defaults preserve the original seasonal 80/10/10 split, 300-epoch limit and patience 30. Do not interpret these experiments as operational leakage-free forecasts.

## How to use

This is a **code-only package**, not a complete standalone training bundle. To run it, supply the original comparison project's `data/` directory (including the station coordinate JSON) and install that project's `requirements.txt` separately. The packaged Python files do not contain either of those resources.

From the extracted package root, after supplying those inputs:

```powershell
python run_all.py --run-id brentwood_2025_seed42 --year 2025
```

Or run one outer wrapper at a time:

```powershell
python .\L\run.py --run-id brentwood_2025_seed42 --year 2025
python .\L.1\run.py --run-id brentwood_2025_seed42 --year 2025
python .\spatial_L.1\run.py --run-id brentwood_2025_seed42 --year 2025
python summarize.py --run-id brentwood_2025_seed42
```

Do not use `source/run.py` as the new-data entry point: those protected originals retain their historical date constants. Run the outer wrappers instead.

The baselines default to CPU because their unchanged validation path retains a CPU tensor. This package does not add a baseline GPU patch. Running the commands above starts training; no training was performed to create this ZIP.

Each experiment saves its own `results/<run-id>/`. After all three complete, the main combined report is `results/<run-id>/comparison.md` at the package root. Previously reported metrics and training artifacts are not included in this archive.

This archive represents the files available in the local comparison project at packaging time. A neighbor-exclusion change made on another PC is not implicitly included unless that changed file was copied into this local project first.

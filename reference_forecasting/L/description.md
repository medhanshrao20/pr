# L baseline — unchanged source, new Brentwood data

Use `python run.py --run-id experiment_name --year 2025` from this folder, or the root `run_all.py`. Local results stay in `results/<run-id>/`.

The source files in `source/` are copied unchanged. The wrapper calls original cleaning, feature construction, alpha selection and training functions. No baseline function/class is replaced. Actual new-data season dates are supplied outside the copied runner's historical period constants. Dataset schema conversion does not modify measurements.

K is fixed at 11/10/11/10 for spring/summer/autumn/winter. Alpha in {25,100,500,2000} minimizes original Ridge validation RMSE. The original full-season VMD–entropy–CA–EWT feature bank feeds Ridge plus KANInformer residual correction. Ridge solves squared training error + 0.1||W||²; residuals use the original MinMaxScaler. Defaults retain 80/10/10 seasonal splits, 300 epochs, patience 30 and seed 42.

Only Brentwood is used for fitting and forecasting. Baseline execution defaults to CPU because the protected source's validation tensor is CPU; no GPU compatibility fix is injected. The root description explains strict-zero MAPE handling and the separate external nonzero-target diagnostic. Source-compatible prediction values are preserved and timestamp/quality-audit columns are appended.

The copied `source/README.md` is historical documentation, not the instructions for this new dataset. Use this description and `../description.md`. Do not directly run the historical source main with its old date constants.

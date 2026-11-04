# Verification — 2026-10-25

Executed `python -m unittest discover -s tests -v`: **5 tests passed in 8.123 seconds**.

Checks:

1. Copied L/L.1 Python files match captured original hashes; original train/metric functions resolve to the copied source paths; original epochs=300 and patience=30 remain.
2. New-data legacy-schema conversion preserves timestamps, 2400 semantics, measurements and original unit conversions; latest covered seasonal year is correctly selected.
3. The spatial Ridge adapter gives the same predictions as own-feature Ridge and ignores arbitrary neighbor-feature changes. The graph neural path has shape B×3, predicts only Brentwood and has gradients through neighbor inputs.
4. Strict MAPE is reported undefined at exact-zero targets; the separate nonzero diagnostic and counts are correct.
5. All three experiment adapters execute end-to-end on temporary **synthetic data**, using unchanged original train functions for one CPU epoch with small test-only model overrides and synthetic feature-bank fixtures (no actual VMD sweep). Checkpoints, forecasts and the final matched comparison are produced. All outputs have target station 47. Modified ground truth is rejected by the comparison collector. Baseline model/Ridge/train objects remain/restored as required.

Source protection was separately checked with SHA-256: all 61 fingerprinted pre-existing files unchanged; all 18 copied source/reference files identical to their originals.

Actual supplied station files were copied and converted into the old CSV schema only. **No actual-data cleaning/decomposition search, Ridge fitting, neural training or forecasting experiment was executed.** No real-data results or performance claims are included. GPU execution is not verified; unchanged baseline validation is explicitly CPU-only in these wrappers.

Synthetic outputs lived in temporary directories and were deleted. Project results directories are empty until the user starts the experiments.

# L.1 baseline — unchanged source, new Brentwood data

Run `python run.py --run-id experiment_name --year 2025`. Outputs remain in this folder's `results/<run-id>/`; no neighboring observations enter the baseline.

`source/` contains unchanged copied L.1 code. The separate wrapper invokes its original cleaning, full-season feature/window construction, parameter selection and train functions, with actual new seasonal dates. No baseline function, model class, metric definition, Ridge implementation or optimizer is replaced.

For a in {25,100,500,2000}, k*(a)=argmin_(k=2..12) mean(|y−sum modes|/max(|y|,1e−12)). Then a*=argmin_a Ridge_validation_RMSE(a,k*(a)). The selected pair is recomputed on the new data. VMD/EWT operate before splitting, retaining the original leakage mechanism. Final prediction is local Ridge + inverse-MinMax-scaled KANInformer residual, using unchanged original training settings (80/10/10 seasonal split, 300 epochs, patience 30, seed 42).

Save the same run ID for Spatial L.1: it can reuse this Brentwood selection only when hashes, year and configuration match, keeping the direct comparison focused on adding graph context. Final comparison verifies identical Brentwood scored targets and times.

CPU is the default to preserve the protected source's validation behavior. Exact-zero strict MAPE remains undefined; a separate explicitly labelled nonzero-target diagnostic is produced externally, not by replacing the original metric. See the root description for complete formulas, outputs and commands.

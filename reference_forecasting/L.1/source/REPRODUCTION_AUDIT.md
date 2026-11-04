# Reconstruction-K / prediction-alpha variant

## Provenance and execution status

Source: the sibling final `original_dataleak` folder under
`journal/kan_original/kan_original`. The older folder under
`projects all mixed/kaninformer_original` was not used.

This is a separate experiment with pipeline name `original_dataleak.v1`.
No decomposition, model checks, ridge fitting, neural training, or forecasting
was executed during code preparation. Selected parameters and scores do not yet
exist for this variant. No baseline results or checkpoints were copied.

The source audit is preserved as `reference/BASELINE_REPRODUCTION_AUDIT.md`.
Its historical verification/result claims concern the original implementation,
not execution or validation of this variant.

## Scientific change

The original final code fixes K to the paper's 11/10/11/10 values and selects
alpha by ridge validation prediction RMSE. This variant exhausts K=2..12 for
each alpha in 25/100/500/2000. It chooses minimum full-season VMD RER separately
for each alpha, then chooses the resulting pair using raw-unit ridge validation
RMSE across all three horizons. The 3% threshold has no selection role.

RER is the mean absolute relative reconstruction residual, not a squared-energy
ratio. The denominator floor of 1e-12, mode ordering, and odd-length padding
follow the baseline. The final decomposition RER is checked against the selected
candidate's RER before neural training. Incomplete/nonfinite candidate searches
abort instead of selecting from a shortened grid. Exact ties are deterministic.

This is a local hybrid selection experiment. It does not reproduce the paper's
threshold/stabilization K-selection procedure, and the paper does not document
this ridge-based alpha-selection strategy. Minimum RER does not guarantee
distinct physical modes or best forecast accuracy.

## Preserved settings and limitations

The original final model, CA/EWT processing, expanded bank, input scaling,
residual scaling, training loop, and metrics remain in use. Variable K changes
the feature count, so fitted ridge/projection dimensions can differ. Neither
existing checkpoint files nor previous forecast errors apply to the new run.

Full-season cleaning and decomposition continue to expose each forecast's
features to later seasonal observations. Validation targets score alpha choices,
while test targets are not used in that score or ridge fitting. This does not
remove the test interval's contribution to full-season transformations.

Autumn and winter coverage remain incomplete in the supplied raw data. ET field
terminology, spring computed-PCC disagreement, unpublished decomposition
settings, paper-versus-author-source conflicts, missing baseline/SHAP/MWC
experiments, and unmatched targets/settings remain as recorded in the source
audit. Exact numerical replication and component-wise superiority are not claimed.

## Validation boundary

Only static source review and file-provenance comparisons were allowed for this
task. New mocked selection checks are supplied for later execution. No passing
runtime tests, training completion, selected K/alpha, new forecasting score, or
GPU verification is asserted.

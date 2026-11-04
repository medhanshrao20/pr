# Protected-code integrity — 2026-10-25

* **61 pre-existing files** fingerprinted before creating this comparison project were rechecked after implementation. All SHA-256 values matched. This includes the original L/L.1 source files, selected original data/documentation, and the existing spatial project code. No pre-existing code file was edited.
* **18 copied source/documentation/reference files** in `L/source/` and `L.1/source/` matched their original files byte for byte. Per-file hashes are recorded in `copied_source_hashes.json`.
* The original historical CSV was not copied into the new baseline data paths. Both new `source/data/raw/hourly.csv` files contain the same new Brentwood measurements in the legacy schema. Conversion preserves timestamps and source units; original loading converts units as before.
* Baseline L and L.1 use original function and model class objects. No baseline monkey-patching, optimizer changes, residual-scaler replacement, feature-bank replacement or CUDA patch is applied.
* The new spatial experiment alone supplies its additional graph model and Brentwood-only Ridge input adapter to its separate loaded module. Its source snapshot remains unchanged.
* Date selection, data-schema conversion, output audits and final comparison are separate files in this new project. Inherited baseline CPU-validation behavior and exact-zero MAPE behavior are explicitly documented rather than silently fixed in protected code.

The fingerprint manifest stores historical absolute paths as provenance, not runtime dependencies. The copied sources and data are local to this comparison folder.

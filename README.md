# WAFER-GUARD: Audited & Repaired Excursion Detector

GDG Tech Wildcard Entry (AI/ML) submission.

- `wafer_guard_fixed.ipynb` is the repaired PyTorch pipeline. It reproduces each defect (Section A), then trains and evaluates the corrected supervised autoencoder at natural defect prevalence.
- `AUDIT.md` contains the findings, the fixes and the true measured metrics.

Run: put `secom_telemetry_modified.csv` next to the notebook and run all cells (about 1 minute on CPU).

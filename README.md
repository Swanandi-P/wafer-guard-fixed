WAFER-GUARD: Audited & Repaired Wafer Anomaly Detector

GDG Tech Wildcard Entry · AI / Machine Learning An audit and repair of a PyTorch supervised-autoencoder pipeline that detects defective semiconductor wafers (SECOM telemetry, Fab-7).

The original notebook reported F1 = 0.99 and passed its release gate. This audit shows that the score came from data leakage, a silent tensor-broadcasting bug, and mishandled legacy-tool telemetry. After the repairs, the honest result at the natural defect rate is F1 ≈ 0.12, so the recommendation is NO-GO for release.

Results at a glance

	                          Original notebook	         Repaired notebook
Test-set defect rate	 50 % (artificially balanced)	5.4 % (natural, untouched)
Split	                           Random,            	Chronological (past → future)
F1	                               0.9915 ❌ invalid	0.129 (5-seed mean 0.117 ± 0.043)
PR-AUC (AP)	                       1.000 ❌ invalid	0.075 (5-seed mean 0.097, no-skill = 0.054)
ROC-AUC	                           1.000 ❌ invalid	0.579 (5-seed mean 0.594)
Release gate (F1 ≥ 0.90)	           Passed	              Failed: NO-GO

All numbers are produced by the notebook itself. Full details are in AUDIT.md.

The three silent bugs

#	Bug	What went wrong	Fix
1	Data leakage	SMOTE, the imputer and the scaler were fit on all data before a random train/test split. 274 of 293 test "defects" were synthetic near-copies of training rows.	Chronological 60/20/20 split. Every preprocessing step is fit on the training window only, and imbalance is handled with pos_weight in the loss.

2	Corrupted loss gradients	Input [B,1,F] minus reconstruction [B,F] silently broadcast to [B,B,F], so the autoencoder learned the batch average instead of each wafer.	Inputs kept as [B,F], with shape asserts on every loss term.

3	Invariant blindness (Probe Station PS-L)	sensor_0 = 0.0 (sentinel) and sensor_1 ∈ {100, 200, 300} (firmware codes) were imputed and scaled as real measurements, distorting both sensors.	An is_legacy flag plus one-hot firmware-code columns. Legacy values are masked out of continuous scaling and never overwritten.

Secondary fixes: a validation-tuned decision threshold (instead of a fixed 0.5), early stopping on validation PR-AUC, removal of train-constant sensors, and outlier clipping.


Repository structure

├── wafer_guard_fixed.ipynb      
Repaired pipeline, with audit evidence in Section A

├── AUDIT.md                      
Findings ranked by severity, metrics, residual risks

├── secom_telemetry_modified.csv  
Dataset (1,567 wafer lots × 590 sensors)

└── README.md                      
Notebook walkthrough


Section	Contents
A · Audit evidence	Reproduces each defect with numbers: leakage, the broadcast shape, and the legacy-sensor distortion

2 · Preprocessing	Causal, train-only WaferPreprocessor with legacy-station handling

3–4 · Model & training	Supervised autoencoder with shape guards, class-weighted BCE and early stopping

5 · Evaluation	Threshold chosen on validation, single evaluation on the future test window

6 · Robustness	5 seeds, a SMOTE-on-train-only ablation and a logistic-regression baseline

7 · Release gate	Honest verdict

How to run

bash
pip install
"torch>=2.2" 
"scikit-learn>=1.2"
"imbalanced-learn>=0.11"
pandas numpy joblib
jupyter notebook wafer_guard_fixed.ipynb  
# Run All (about 1 min on CPU)

It also runs on Google Colab: upload the notebook and the CSV, then use Runtime → Run all.

Why the score dropped and why that's correct

At a 6.6 % defect rate, a model can look excellent on a balanced or leaked test set while being nearly useless on the line. PR-AUC compared with its no-skill baseline and F1 at a validation-chosen threshold show how the detector would actually behave in production. The repaired model carries a weak but real signal, and it does not yet justify deployment.

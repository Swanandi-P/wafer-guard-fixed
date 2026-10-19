# WAFER-GUARD Audit: Fab-7 Excursion Detector

**Recommendation: NO-GO.** The reported F1 = 0.99 is an artefact of leakage and a balanced test set. At natural prevalence the repaired model scores **F1 ≈ 0.12** and **PR-AUC ≈ 0.10**, against a no-skill baseline of 0.054. It detects defects only weakly better than chance.

All numbers below are produced by `wafer_guard_fixed.ipynb` (Section A reproduces each defect; seed 1337, CPU).

## Findings, ranked by severity

### 1. Data leakage: invalidates the metric (Critical)
**Location:** cell *2 · Preprocessing*. The imputer, scaler and SMOTE are fit on all 1,567 rows, and the train/test split happens *after* SMOTE.
**Mechanism:** SMOTE synthesises defects by interpolating between real defects. Splitting afterwards places synthetic neighbours of training defects in the test set. The random split also ignores time, and the imputer and scaler see future and test statistics.
**Evidence:**
- Test prevalence is 50.0 % (natural 6.6 %).
- 274 of 293 test "defects" are synthetic.
- The median distance to the nearest training row is 2.4 for synthetic test defects, versus 17.8 for real test passes.
- 99.7 % of test lots are older than the newest training lot.

**Fix:** Sort by `timestamp` and split chronologically (60 / 20 / 20 for train / val / test). Fit the preprocessor on train only. Handle imbalance inside training with `pos_weight` (SMOTE on train only was tested as an ablation and was no better). Evaluate once on the untouched future window.

### 2. Broadcasting in the reconstruction loss: corrupted gradients (High)
**Location:** `make_loader` (`unsqueeze(1)` → `[B,1,F]`) combined with `forward` (`flatten` → output `[B,F]`) in `recon_loss`.
**Mechanism:** `[B,F] − [B,1,F]` silently broadcasts to `[B,B,F]`. Every reconstruction is scored against every wafer in the batch, so the optimum is the batch mean, not the input.
**Evidence:**
- The difference tensor is `(64, 64, 590)`.
- On a random batch, a *perfect* reconstruction scores 1.96, while predicting the batch mean scores 0.98.
- The original recon loss plateaued at about 0.68. After the fix it keeps falling (0.57 at early stop).

**Fix:** The loader yields `[N,F]`. `forward` asserts a 2-D input, and the training loop asserts `recon.shape == x.shape` and `logits.shape == y.shape`.

### 3. Invariant blindness: legacy probe station PS-L (High)
**Location:** Data and preprocessing. Nothing in the pipeline distinguishes the 146 legacy wafers (9.3 %).
**Mechanism:** `sensor_0 = 0.0` is a sentinel, and `sensor_1 ∈ {100, 200, 300}` are firmware codes. Both were mean-imputed and z-scaled as real measurements.
**Evidence:**
- Imputer mean for `sensor_1` is 2281 (true 2496), and for `sensor_0` it is 2732 (true 3014).
- Scaler std for `sensor_1` is 674 (true 80), which squashes genuine variation by about 8×.
- Legacy defect rate is 0 %, versus 7.3 % for normal wafers, so the flag itself carries information that the old pipeline destroyed.

**Fix:** Add an `is_legacy` flag and one-hot `fw_code_100/200/300`. For legacy rows, `sensor_0` and `sensor_1` are masked to missing *before* scaling, so they never enter the scaler statistics. They are then held at a neutral 0, with the flag explaining why. Codes are never scaled or overwritten. Unknown codes raise a warning.

### 4. Secondary defects (Medium)
- Fixed 0.5 threshold. **Fix:** tune the threshold for max F1 on the validation window and freeze it before test.
- No validation set or early stopping. **Fix:** early stopping on validation AP, patience 10.
- 122 train-constant sensors were scaled. **Fix:** drop them (fit on train).
- No outlier guard. **Fix:** clip at ±10 σ.

## True metrics at natural prevalence
Test window: most recent 314 lots (2008-10-02 → 10-17), with 17 defects (5.4 %, the period's natural rate; not resampled).

| Model | F1 | Precision | Recall | ROC-AUC | PR-AUC (AP) |
|---|---|---|---|---|---|
| **Repaired SAE, seed 1337** | **0.129** | 0.075 | 0.471 | 0.579 | 0.075 |
| Repaired SAE, 5 seeds (mean ± sd) | 0.117 ± 0.043 | 0.098 ± 0.062 | 0.353 ± 0.208 | 0.594 ± 0.096 | 0.097 ± 0.037 |
| SAE + SMOTE (train only), 5 seeds | 0.107 ± 0.061 | 0.062 ± 0.036 | 0.424 ± 0.289 | 0.590 ± 0.068 | 0.088 ± 0.021 |
| Logistic regression baseline | 0.163 | 0.093 | 0.647 | 0.641 | 0.091 |
| *No-skill baseline* | — | — | — | 0.5 | 0.054 |

Seed-1337 confusion: TN = 198, FP = 99, FN = 9, TP = 8.

**Metric choice:** At about 6 % prevalence, accuracy and ROC-AUC flatter the model. PR-AUC (against its prevalence baseline) and F1 at a validation-chosen threshold reflect the real precision/recall trade-off at the line.

## Residual risks
- Only 17 test and 11 validation defects, so the threshold and metrics have high variance (see the seed spread).
- Recipe and tool drift: defect rates move from 8.1 % to 3.5 % to 5.4 % across the windows. The model needs rolling retraining and monitoring.
- The supervised autoencoder does not beat a regularised linear baseline. Feature selection and simpler models should be explored before any release.

# Per-prediction uncertainty — what validated, and what didn't

Goal: replace the global "typical error" with a **molecule-specific** one. The rule applied
throughout: a per-molecule number ships only if it was **shown to predict the actual error**.
A number that varies per molecule but doesn't track error is worse than an honest average.

```bash
python validation/calibrate_uncertainty.py     # regenerates uncertainty_calibration.json
```

## Signals tested

| Model type | Signal | Why |
|---|---|---|
| RandomForest | std across the individual trees | independent ensemble → a real variance estimate |
| XGBoost / LGBM | std of **staged** predictions (40→100 % of boosting rounds) | boosted trees are additive corrections, so *per-tree* std is meaningless; how much the prediction is still moving is the usable proxy |

## Acceptance gate (both required)

- Spearman ρ(spread, \|error\|) ≥ **0.30**, and
- effect size: top-spread quartile error ≥ **1.5×** bottom-spread quartile error.

The second criterion matters: **LOI scored ρ = +0.40 but its quartile errors were flat**
(0.539 → 0.55, a 1.02× ratio). The ordering was right, the information content nil. ρ alone
would have shipped a useless number.

## Results (n≈400 per property, in-distribution)

| Property | method | ρ | Q4/Q1 | ships |
|---|---|---|---|---|
| **Refractive** | ensemble | **+0.946** | **6.04×** | ✅ |
| **Solubility** | staged | **+0.569** | **2.61×** | ✅ |
| Td | staged | +0.298 | 1.94× | ✗ |
| BandgapBulk | staged | +0.376 | 1.41× | ✗ |
| Tg | staged | +0.252 | 2.17× | ✗ |
| BandgapChain | staged | +0.251 | 1.75× | ✗ |
| Tm | staged | +0.156 | 1.30× | ✗ |
| LOI | staged | +0.404 | 1.02× | ✗ (flat) |
| ThermalCond | staged | +0.014 | 1.24× | ✗ |
| CTE | staged | −0.134 | 1.02× | ✗ |
| EPS | staged | +0.024 | 0.45× | ✗ |

**2 of 16 properties earned a molecule-specific error bar.** The other 14 keep the global
golden-set MAE — clearly labelled as such. In the UI a molecule-level value is marked `＊`.

### A trap worth recording
Measured against the **golden set**, RF spread appeared *anti*-correlated with error (ρ = −0.36),
which would have killed the feature. That was a confound: out-of-domain, the model reverts to the
mean, so the error is dominated by *systematic bias*, which no spread signal can see. Re-measured
**in-distribution** the same signal scored ρ = **+0.93**. Uncertainty signals describe *noise*,
not *bias* — they must be validated in-distribution, and the applicability-domain banner remains
the thing that warns about bias.

## How the number is produced

`prop_error_for(prop, smiles)` maps the measured spread through the calibration quartiles onto a
bounded multiplier (**0.45× – 2.0×**) of the property's global MAE. A settled prediction tightens
the bar, an unsettled one widens it; the well-validated global MAE stays the anchor, so the value
can never run away. Properties without a calibration return the global MAE unchanged.

## Related: the applicability-domain penalty (opt-in)

`DOMAIN_PENALTY_ON` (sidebar: *"Stay in the model's confidence zone"*, **off by default**) adds a
GA cost for candidates far from the training set. It is zero while a candidate is still reasonably
similar to known polymers — ordinary novelty is not punished — and only bites past the `AD_WARN`
edge where the models extrapolate. Off by default because it trades novelty for trustworthiness,
which is the wrong default for a tool built to find *novel* polymers.

---

## Update: bagged companion ensembles (the fix for the other properties)

The table above shows why most properties failed: a single XGBoost has no valid internal
uncertainty. The fix is to *create* one — `validation/train_bagged.py` trains **8 independently
seeded models on bootstrap resamples** per property. The spread across members is a real variance
estimate, exactly like a RandomForest's.

```bash
python validation/train_bagged.py          # train the bags (~15 min)
python validation/compare_bagged.py        # fair golden-set comparison vs the deployed models
```

### Calibrated on a true hold-out
Unlike the first pass (in-sample only), these are calibrated on a **20 % hold-out**, which is the
statistically correct basis:

| property | ρ (held-out) | Q4/Q1 | ships |
|---|---|---|---|
| Hansen | +0.68 | 6.31× | ✅ |
| BandgapBulk | +0.52 | 3.43× | ✅ |
| BandgapChain | +0.55 | 3.65× | ✅ |
| Tm | +0.57 | 2.70× | ✅ |
| Td | +0.55 | 2.31× | ✅ |
| BandgapCrystal | +0.39 | 2.02× | ✅ |
| Degradability | +0.74 | 2.81× | ✗ (see below) |

### Companion, not replacement — and why
On the **golden set** (out-of-sample for *both* models, so the only fair comparison) bagging did
**not** improve accuracy: it won CTE (32.5→23.2) and ThermalCond, lost Tg (44.0→46.7) and
Solubility, tied Tm. Bagging adds little accuracy on top of XGBoost, which already subsamples
internally.

So the bags are used as **uncertainty companions**: predictions still come from the deployed
models (**zero accuracy risk**), and the bag is consulted only for spread. This is valid because
the bag's spread was verified to predict the *deployed* model's error too — 6 of 7 properties
transferred (ρ 0.30–0.41). **Degradability did not transfer** (ρ=0.18) and is therefore excluded,
despite having the best bag-internal correlation.

**Result: 8 properties now carry a molecule-specific error bar** (Td, Tm, BandgapBulk,
BandgapChain, BandgapCrystal, Hansen via bags; Refractive, Solubility via internal signals) —
up from 2. The rest keep the global measured MAE.

Controlled by `USE_BAGGED_UNCERTAINTY` in `appv3.py` (adds ~43 MB of lazily-loaded models; set
`False` for a lightweight deploy and those properties fall back to the global MAE).

### Side benefit: better global error bars
The hold-out runs also produced genuine out-of-sample MAEs, which replaced optimistic in-sample
figures in `PROP_MAE` — most notably **Td: 64 → 45.3** — and added Hansen and Degradability, which
previously had no error bar at all. Sources are labelled `gold` (golden set) > `held` (hold-out) >
`in` (in-sample), and the more conservative is preferred where both exist.

### Gas permeability — unit convention resolved
`GasPerma` was previously excluded because the HF `Gas_Permeability` split is ~1e-13 SI while the
model outputs Barrer. Investigating it:

* the **Barrer convention is correct** — the model was trained on the PolymerGasMembraneML
  benchmark (`datasetA_imputed_all.csv`), **CO₂ column, log10(Barrer)**, median 26.6 Barrer;
* the **HF split is unusable as a target**: all 724 rows are PoLyInfo with **no gas label**, so it
  mixes He/H₂/O₂/N₂/CO₂/CH₄ — which explains the 1.7-decade offset against a CO₂-only model
  (rank correlation was still +0.62, i.e. the model was fine, the comparison was not).

The bag was therefore trained on the original CO₂ source: **hold-out 3.28× fold error** (matching
the 3.6× reported at training), bag beats single (R² 0.784 → 0.810), uncertainty validates
(ρ=+0.40, 3.50×) and transfers to the deployed model (ρ=+0.30). Because the target is log-scaled,
its error is reported as a **fold factor** (`×/÷`), and the per-molecule scaling is applied in log
space — a 3.3× base at the maximum multiplier becomes 10.9×, not 6.6×.

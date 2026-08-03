# van Krevelen group-contribution estimator (`van_krevelen.py`)

An independent, structure-only second opinion on a polymer's predictions — the polymer-science
analog of xtb for the **bulk** properties. It needs no external tool (pure RDKit), so it runs
everywhere the app runs, including Streamlit Cloud.

```bash
python van_krevelen.py        # prints the validation table below
```

`vk_estimate(smiles)` tiles the repeat unit into groups (SMARTS, functional groups first) and
sums Hoftyzer–Van Krevelen contributions. `vk_verify(smiles, model_preds)` returns ✅/⚠️/❌
checks in the app's verification panel.

## What it estimates, and how accurate (validated on 14 canonical polymers)

| Property | Method | MAE | Notes |
|---|---|---|---|
| **Refractive index** | molar refraction (RDKit MolMR) / molar volume, Lorentz–Lorenz | **0.064** | beats the ML model (0.18) **and** varies with structure, where the model reverts to ~1.68 |
| **Solubility** (Hildebrand δ) | HvK cohesion Fd/Fp/Eh | **1.16** (cal/cm³)^½ | independent physics estimate; weakest on strong H-bonders |
| **Hansen** (total δ) | same, in MPa^½ | ~2.4 MPa^½ | a genuinely new independent check (no ML baseline) |
| density | M / V (group molar volume) | 0.09 g/cm³ | bonus (not a model output) |

Coverage is reported per polymer (fraction of heavy atoms assigned to a group); < 100 % means an
unrecognized fragment and the check is softened rather than hard-failed.

## What it deliberately does NOT estimate

**Tg, Tm, Td, CTE, thermal conductivity, gas permeability.** Van Krevelen predicts these from
tabulated Yg/Ym (and other) group functions. Without the authoritative tables, a from-literature
calibration on ~26 polymers **overfits**: leave-one-out MAE was **~98 °C (Tg) / ~129 °C (Tm)** —
*worse* than the trained model's own out-of-sample ~44 °C. Shipping that would mislead. So for
these properties:

- Trust the **model's** value with the golden-set error bar (`GOLDEN_SET_VALIDATION.md`:
  Tg/Tm direction solid, absolute ±~45 °C; LOI ±3 %; Solubility ±0.8).
- For a rigorous per-polymer number, use **MD (RadonPy)**.
- To extend this estimator, add the real van Krevelen Yg/Ym group functions to the `GROUPS` table.

## In the app

The results-tab **🔬 Verify this result** panel merges, per polymer:
`BandgapChain` + `EPS` (xtb, when installed) and `Refractive` + `Solubility` + `Hansen`
(van Krevelen, always). Each card shows the model value, the independent estimate ± its
validated uncertainty, a ✅/⚠️/❌ verdict, and which method produced it.

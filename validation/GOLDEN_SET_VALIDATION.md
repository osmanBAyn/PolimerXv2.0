# Out-of-Sample Validation on Canonical Polymers (Golden Set)

For the properties **xtb cannot verify per-polymer** (the bulk / thermal / transport ones),
this is the honest confidence measure: predict 18 textbook polymers with known handbook
values and measure the error. Held-out ML R² is optimistic (same data distribution); *named*
polymers test whether a number you'd actually trust is trustworthy.

```bash
python validation/golden_set.py
```

Reference values are approximate handbook figures (Polymer Handbook / van Krevelen / CROW /
Bicerano). **MAE = the typical error to expect on a new polymer for that property.**

## Results (18 canonical polymers)

| Property | n | MAE | Spearman | Verdict |
|---|---|---|---|---|
| **LOI** | 11 | **3.3 %** | **+0.91** | ✅ trust the absolute value |
| **Solubility** (Hildebrand δ) | 17 | **0.84** (cal/cm³)^½ | **+0.79** | ✅ good; weakest on strong H-bonders (Nylon-6, PVA) |
| **Tm** | 12 | 45 °C | +0.80 | 🟡 trust the ranking; absolute ±45 °C |
| **Tg** | 17 | 44 °C | +0.74 | 🟡 trust the ranking; absolute ±44 °C |
| **Refractive index** | 18 | 0.18 | +0.68 | 🔴 collapses to ~1.68 on canonical polymers |
| **EPS** (dielectric) | 18 | 1.04 | +0.14 | 🔴 collapses to ~3.3; poor ranking here |

## The key nuance: canonical polymers are *out of distribution*

The models were trained on a specific dataset. **GA outputs resemble that training
distribution; textbook polymers (PE, PTFE, PDMS…) often do not.** So this table is a
*worst-case, out-of-distribution* test, and the split it reveals is important:

- **LOI, Solubility, Tg, Tm generalize** — they hold up even on out-of-distribution polymers,
  so you can trust these predictions for an optimal GA polymer within the stated error bars
  (LOI ±3 %, Solubility ±0.8, Tg/Tm direction solid, absolute ±~45).
- **Refractive & EPS only work *inside* the training distribution.** On canonical polymers
  they revert to the mean (RI ≈ 1.68 for everything; ε ≈ 3.3, missing PVDF's 9.0). For
  GA-generated polymers (in-distribution) the xtb polarizability check is the more relevant
  evidence — there they ranked at Spearman +0.83 / +0.88 (see XTB_VALIDATION.md). **Treat
  their absolute values with caution whenever the polymer looks unusual.**

## How to use this when verifying an optimal polymer

| Predicted property | How to trust the number |
|---|---|
| Band gap (chain/bulk/crystal) | **Verify per-polymer with xtb** in the app (±0.9 eV, calibrated) |
| LOI | Trust absolute (±3 %) |
| Solubility / Hansen | Trust absolute (±0.8); softer for strong H-bonders |
| Tg, Tm, Td | Trust the direction; absolute ±~45 °C |
| Refractive, EPS | Reliable only for training-like structures; sanity-check RI/ε against the app's xtb polarizability readout |
| GasPerma, ThermalCond, CTE, Crystallinity, Recyclability, Degradability | No independent per-polymer oracle yet — see roadmap below |

## To push further (independent per-polymer estimates for the rest)

- **Group-contribution (van Krevelen / Bicerano)** — independent absolute estimates from SMILES
  for Tg, Tm, Td, CTE, thermal conductivity, solubility/Hansen, density, gas permeability. The
  polymer-science analog of xtb; implementable as a SMARTS-group summation.
- **RadonPy (MD)** — gold-standard bulk-property simulation (Tg, density, CTE, thermal
  conductivity, RI, solubility). Hours–days per polymer; best reserved for a final candidate.
- **Cross-model consensus** — Polymer Genome, polyBERT for a second ML opinion.

_Values here are approximate handbook figures; extend the `GOLD` table in `golden_set.py`
with your own trusted references to sharpen the numbers._

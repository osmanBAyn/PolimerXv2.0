# Independent Quantum Validation of Polsen's Electronic-Property Models

**Method:** GFN2-xTB (semi-empirical quantum chemistry, via `xtb`), used as an *independent*
check the ML models never saw during training. Reproduce with:

```bash
export XTB_EXE=/path/to/xtb        # conda install -c conda-forge xtb
python validation/xtb_crosscheck.py all -n 24
```

Polymers are real repeat units sampled from `OsBaran/Polimer-Ozellik-Tahmini`, stratified to
span each property's range. **Spearman rank correlation is the headline metric**, because a
semi-empirical method has a large, systematic *absolute* offset from the DFT the models were
trained on — but if xtb and the model **order** polymers the same way, the model's ranking is
physically sound.

---

## Results

| Property | Model | Independent xtb check | **Spearman (model vs xtb)** |
|---|---|---|---|
| **Band gap (chain)** | `xgb_band gap chain` | HOMO–LUMO gap of n=1,2,3 oligomers, extrapolated to ∞-chain | **+0.82** |
| **Refractive index** | `rf_refractive_index_v2` | Lorentz–Lorenz vs electronic polarizability density α/V | **+0.83** |
| **Dielectric (EPS)** | `xgb_eps_v2` | Clausius–Mossotti vs electronic polarizability density α/V | **+0.88** |

All three models' rankings are independently confirmed by an unrelated quantum method. ✅

### Band gap (n=24, span 0.02–9.86 eV)
```
model vs dataset : Spearman +0.99   R2 +0.98   MAE 0.20 eV   (in-sample fidelity)
xtb   vs dataset : Spearman +0.86              MAE 2.12 eV   (xtb tracks DFT ranking)
model vs xtb     : Spearman +0.82              MAE 2.08 eV   (INDEPENDENT validation)
```
The ~2 eV MAE is the **expected** GFN2-xTB-vs-DFT systematic offset, not a model error — hence
rank correlation is the fair metric. Side-finding: one saturated aliphatic repeat unit is
labelled 6.3 eV in the dataset but xtb gives ~11 eV (physically more reasonable for a fully
saturated chain), suggesting a possible mislabel in that dataset row.

### Refractive index (n=22, physical subset RI ≤ 1.8)
```
model RI (Lorentz-Lorenz) vs xtb alpha/V : Spearman +0.83   (INDEPENDENT validation)
dataset RI                vs xtb alpha/V : Spearman +0.77
model RI                  vs dataset RI  : Spearman +0.99
```
The model correlates with the xtb physics **better than the raw dataset does** (0.83 vs 0.77):
it preserved the true polarizability trend while smoothing dataset noise.

### Dielectric / EPS (n=18)
```
model EPS (Clausius-Mossotti) vs xtb alpha/V : Spearman +0.88   (INDEPENDENT validation)
```
**Caveat:** xtb α is the *electronic* polarizability, which maps to the high-frequency limit
ε∞ ≈ n². Static ε also carries a dipolar-orientation term, so for very polar polymers the model
may legitimately sit above the electronic-only trend. A +0.88 rank correlation confirms the
electronic backbone of the model's ordering is correct.

---

## What can and cannot be quantum-verified this way

**Verifiable (electronic-structure properties):** band gap, refractive index, dielectric
constant — all done above, all pass.

**Not verifiable with a single-molecule semi-empirical calc:** Tg, Td, Tm, LOI, gas
permeability, solubility/Hansen, thermal conductivity, CTE, crystallinity, recyclability,
degradability. These depend on bulk chain packing, morphology, thermal history and kinetics —
not on an oligomer's electronic structure — so xtb cannot serve as an independent oracle for
them. They remain validated by held-out ML metrics only (see the reliability tiers in `appv3`).

**Deployment note:** `xtb` is an *offline validation tool*. The Streamlit app does not call it
and does not depend on it — these numbers are standing evidence about model trustworthiness.

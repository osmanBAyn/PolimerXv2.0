# Full application test — 6 real design missions

End-to-end test of the whole pipeline (GA → predictions → uncertainty → applicability domain →
retrosynthesis → verification), using genuinely hard materials-science targets rather than toy
ones. Each result was then judged by hand for chemical plausibility.

Harness: `scratchpad/full_app_test.py` (25 generations, pop 120, seed 11, 15 seed polymers).

## Results

| mission | result | chemistry verdict |
|---|---|---|
| **Aerospace high-temp** | `*OC1=CC=C(C(=O)C2=CC=C(*)C=C2)C(naphthyl-Cl)=C1` | ✅ **excellent** — a poly(aryl ether ketone), the PEEK/PEKK family, with a chloronaphthyl pendant |
| **High-index optical** | `*COC(=O)C1=CC=C(*)C=C1` | ✅ **real polymer**; monomer = 4-(hydroxymethyl)benzoic acid, **PubChem CID 76360**, commercially available |
| **Low-k dielectric** | `*C1=CC=C2C=CC(C(F)F)=C(*)C2=C1F` | ✅ **sound** — fluorinated poly(naphthalene); retro route (dibromoarene + Yamamoto/Suzuki) is textbook-correct |
| **Organic semiconductor** | fused aromatic imide + biphenyl, gap 1.79 eV | ✅ sensible donor–acceptor design for a 1.8 eV target |
| **CO₂ membrane** | poly(benzylic-arylene), many phenyls | ⚠️ valid but "phenyl salad"; **no retro route**; missed target (473 vs 800 Barrer) |
| **Biodegradable** | `*OCC(*)(CO)CC(=O)O` | ❌ **dubious** — pendant –COOH *and* –CH₂OH on a polyether would self-condense/crosslink |

**Structural checks: 6/6** charge-neutral, chemically sane, valid polymer SMILES.
**Retrosynthesis: 5/6** produced a *verified* route; the 6th is the known cross-ring benzylic
case with no unique monomer (honest no-route).

## The verification layer earned its keep

van Krevelen independently flagged **both** problem cases without being told:

* the biodegradable candidate — Hansen `model 22.2 vs 32.9` **[bad]**, Solubility **[warn]**
  (correctly spotting that a molecule with –COOH + 2×–OH must be far more polar than predicted);
* the refractive index — flagged `warn`/`bad` on **every single mission**.

That second flag turned out to be a real, systematic model defect ⤵

## ❗ Critical finding: the Refractive model cannot discriminate

Predicted refractive index on canonical polymers with known literature values:

| polymer | literature | ML model | error | van Krevelen | error |
|---|---|---|---|---|---|
| PTFE | 1.350 | 1.663 | **+0.313** | 1.336 | −0.014 |
| PDMS | 1.430 | 1.682 | **+0.252** | 1.464 | +0.034 |
| PE | 1.510 | 1.681 | +0.171 | 1.504 | −0.006 |
| PMMA | 1.490 | 1.683 | +0.193 | 1.519 | +0.029 |
| PS | 1.590 | 1.679 | +0.089 | 1.635 | +0.045 |
| PC | 1.585 | 1.715 | +0.130 | 1.539 | −0.046 |
| PSU | 1.630 | 1.733 | +0.103 | 1.568 | −0.062 |
| | | **MAE 0.179** | | **MAE 0.034** | |
| | | ρ **+0.54** | | ρ **+0.93** | |

The model's entire output range is **0.072** across polymers whose true refractive indices span
**0.28** — it returns ≈1.70 for everything, from a fluoropolymer to a polysulfone.
**van Krevelen is 5.3× more accurate.**

Consequences:
1. **Targeting `Refractive` in the GA is currently futile** — the search cannot optimise a
   property the model does not vary.
2. The optical mission's apparent success (predicted 1.72 = target 1.72) is **meaningless**; the
   true value for that ester is ≈1.56.
3. Its per-molecule uncertainty (ρ=0.95) was measured *in-distribution* and gives false
   confidence out-of-domain — a concrete instance of the "noise vs bias" trap in
   [UNCERTAINTY.md](UNCERTAINTY.md).

**Recommended fix:** use `van_krevelen.vk_estimate()["Refractive"]` as the refractive-index
predictor (it is already in the app, needs no new dependency, and is validated at MAE 0.034 here
and 0.064 on the 14-polymer set in [VAN_KREVELEN.md](VAN_KREVELEN.md)), or demote the property to
ranking-only with a hard warning.

## Other observations

* **All six results were flagged out-of-domain** (similarity 0.08–0.28). Partly an artefact of
  seeding with only 15 polymers, but it means every absolute value carries the extrapolation
  caveat — which the app does display.
* **Most proposed monomers are novel** (not in PubChem). Expected when designing novel polymers;
  the app now offers a PubChem structure-search link for these.
* **Runner-up candidates proved their worth**: in several missions a runner-up had a materially
  lower SA score than the winner at nearly the same error.


---

## Follow-up: the fix was applied and re-tested (2026-08-01)

`Refractive` is now computed by **van Krevelen group contribution** (`ANALYTIC_PROPS` in
`appv3.py`); the ML model remains loaded only as a fallback. Downstream: reliability tier
`medium → high`, error bar `0.18 → 0.034`, and the ML model's per-molecule uncertainty was
**removed** (its internal spread no longer describes the value we report), as was van Krevelen's
now-meaningless self-verification of the property.

**Accuracy on canonical polymers: MAE 0.179 → 0.034, Spearman +0.54 → +0.93, and the output
range went from 0.072 (near-constant) to 0.299 (real discrimination).**

### The decisive test: can the GA now optimise refractive index?

| target | achieved | structure the GA found |
|---|---|---|
| 1.70 | 1.679 | aromatic, polystyrene-like |
| **1.38** | **1.367** | `*COC(=O)C1=CC(F)=C(C(F)(F)F)C=C1C(*)F` — **fluorinated** |
| 1.45 | 1.441 | partially fluorinated |

The GA independently discovered that **fluorination lowers refractive index** (correct: C–F has
low polarizability) and that **bromine/aromatics raise it**. This was impossible before, when the
model returned ≈1.70 regardless of structure.

### Re-run of all six missions

* **Retrosynthesis: 6/6 verified routes** (was 5/6).
* **CO₂ membrane transformed**: now a polycarbonate bearing a **trimethylsilyl** group, hitting
  **836 Barrer** (target 800; previously 473 with *no* route). Bulky Si(CH₃)₃ is exactly the
  free-volume chemistry behind PTMSP, the most permeable polymer known — and it now decomposes
  cleanly to a bisphenol + carbonic acid.
* **Aerospace**: a poly(aryl ether **sulfone**) with a chloronaphthyl group — the PSU/PES family,
  inherently flame-retardant, Tg 237 °C / Td 519 °C / LOI 42.7 %.
* **Optical**: a brominated aromatic polyamide at RI 1.71 — bromine genuinely raises refractive
  index, so this result is now *meaningful* rather than a false positive.

### One limitation found and fixed
van Krevelen **over-estimates for heavily hydroxylated units** — a polyol returned 1.93, which is
unphysical for an organic polymer. The estimate is now gated to **1.25–1.80** (the real span of
organic polymers), outside which it falls back to the ML model. Verified: that polyol now falls
back, PTFE (1.336) and PS (1.635) still use van Krevelen.

Runs remain deterministic under a fixed seed, and all three regression suites pass.

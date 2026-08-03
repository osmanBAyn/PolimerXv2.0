# Blends, and the chemist's review

Two additions driven by chemist feedback: (1) real materials are blends and alloys, not neat
homopolymers; (2) everything shown must be chemically defensible.

```bash
python validation/validate_blends.py     # miscibility + Fox equation against textbook blends
```

---

## 1. Blends (`blends.py`, "🧪 Blend" tab)

Blend any result with a commodity polymer or another candidate. Nothing here is a new ML model —
it is standard polymer physics applied to the component predictions.

### Miscibility screening
Hansen distance between the repeat units,
`Ra = √(4Δδd² + Δδp² + Δδh²)`, with δd/δp/δh from van Krevelen, plus a Flory–Huggins χ estimate.
Thresholds are deliberately much stricter than the polymer–solvent case (Ra ≤ 3 miscible,
≤ 6 borderline) because a high-molar-mass pair has essentially **no entropy of mixing** to help
it — miscibility is the exception, not the rule.

**Measured accuracy: 23/28 (82 %) on textbook blends.**

Correct on 23 of 28 pairs, including PS/PαMS, PET/PBT and PVC/PVAc (miscible), PC/SAN and
PMMA/SAN (borderline), and PS/PMMA, PE/PP, PA6/PP, PLA/PCL, PS/PDMS, PP/PVOH, PAN/PE (immiscible).

The five misses — PS/PPO, PMMA/PVDF, PVC/PVAc, PS/PVME (all miscible, predicted otherwise) and
PET/PA6 (immiscible, predicted miscible) — are the cases where a *specific interaction* decides
miscibility. **Four of the five carry an explicit caveat flag**; PS/PVME (a weak ether
interaction) is the one unflagged miss.

**Most misses are flagged** with a specific-interaction caveat, because that is
exactly what solubility parameters cannot see: PS/PPO is miscible through π–π stacking and
PMMA/PVDF through a dipole interaction, neither of which shows up in δ. This is a known and
documented limitation of the method, not an implementation bug — so the app states it rather than
hiding it, and reports a confidence level with every verdict.

Two structural overrides make the screen less naive:
* **crystallinity** — two semi-crystalline polymers stay phase-separated in the solid however
  well their δ match (PE/PP is the classic case), so a "miscible" verdict is downgraded;
* **specific interactions** — aryl-ether/aromatic (π–π), carbonyl/halogenated carbon (dipole),
  hydroxyl/carbonyl (H-bond) are detected by SMARTS and reported.

### Blend properties — standard rules only, no invented ones

| property | rule |
|---|---|
| **Tg** | **Fox equation** (miscible); two Tgs reported when phase-separated |
| density | additive specific volume |
| refractive index | Lorentz–Lorenz, volume-fraction weighted |
| gas permeability | semi-log additive (miscible) / **Maxwell** dispersed-phase model (immiscible) |
| solubility, Hansen | volume-fraction additive |
| LOI | weight additive — **flagged**, flame retardancy is often non-additive |
| Td | set by the least stable component |
| Tm | **not averaged** — each crystalline phase keeps its own |
| thermal cond., ε, CTE | rule of mixtures, with the series/parallel **bounds** given, since the true value depends on morphology |

**Fox equation validation: mean error 4.0 °C** across five measured blends (PS/PPO at three
compositions, PVC/PMMA, PEO/PMMA).

---

## 2. Chemist's review (`chem_review.py`)

The property models return a number for every structure — including numbers that are physically
meaningless for that polymer. This module states the caveats before a chemist has to.

### Amorphous vs semi-crystalline: **10/10** on canonical polymers
PS, PMMA, PC, PPO → amorphous; PE, POM, PET, PA6, PTFE, PEO → semi-crystalline
(PEEK and PPS also correctly crystalline).

Encoded rules: a ring is a packing-disrupting **pendant** only if none of its atoms lie on the
backbone (so PET/PEEK stay crystalline); **fluorine is not bulky** (PTFE/PVDF crystallise); and
an **alkyl-substituted backbone arene** disrupts packing (which is exactly why PPO is amorphous
while PEEK is not).

This matters because the app was reporting **"Tm = 206 °C" for polystyrene** — a polymer with no
melting point at all. That is now flagged.

### What else it flags
* **Reactive pairs that cannot coexist in a stable repeat unit** — pendant –COOH with –OH
  (self-esterifies), –COOH with –NH (self-amidates), epoxide with a nucleophile, free isocyanate,
  peroxide, acyclic N–N.
* **Tg/Tm consistency** — Tg must be below Tm for a semi-crystalline polymer, plus the
  Boyer–Beaman Tg/Tm ≈ 0.5–0.8 (K) sanity range.
* **Architecture** — >2 connection points means a branch/crosslink junction, so linear-chain
  property predictions will shift once the network forms.
* **Context a chemist will raise anyway** — halogen-derived LOI (effective but being regulated
  out), heavy aromaticity (poor solubility, may not be melt-processable), large repeat units
  (multi-step monomer synthesis).

Notes are **cautions, not rejections**: the GA exists to find novel chemistry, and an unusual
motif is not automatically wrong.

Both features are behind `SHOW_BLENDS` and `SHOW_CHEM_REVIEW` in `appv3.py`, and the chemist's
review is included in the downloadable text report.


---

## Update (2026-08-03): a capping bug found by the expanded test set

Running 28 blends instead of 10 exposed a real chemistry error. Both modules identified
functional groups by replacing `*` with `[H]` — which **invents end groups the polymer does not
have**:

| repeat unit | as `[H]`-capped | invented |
|---|---|---|
| PLA `*OC(C)C(=O)*` | `CC(O)C=O` | a hydroxyl **and** an aldehyde |
| PET | `O=Cc1ccc(C(=O)OCCO)cc1` | a free –OH |
| PEO `*CCO*` | `CCO` (ethanol!) | a free –OH |

Consequences, both now fixed by capping with **carbon** and ignoring matches that touch a cap:

* `blends.specific_interaction()` was reporting spurious "hydroxyl + carbonyl (H-bond)" on
  essentially every polyester and polyether pair (PET/PBT, PLA/PCL, PEEK/PA6 …);
* `chem_review` could have accused a plain polyester of "carrying both –COOH and –OH" — the sort
  of error that would end a conversation with a chemist. Verified afterwards: **9/9 commodity
  polymers are now clean of stability errors**, while the genuine cases (a GA polyol with a real
  pendant acid + alcohol, PVOH's real pendant –OH) still fire.

## App-wide chemistry audit (`validation/chem_audit.py`)

Physical bounds and cross-property consistency, on 15 canonical polymers and 250 fresh GA
structures:

| check | result |
|---|---|
| canonical polymers — bounds & consistency | **0 violations** |
| GA structures — charged repeat units | **0** |
| GA structures — failing the sanity filter | **0** |
| GA structures — physical bound violations (ε ≥ 1, RI 1.2–2.0, gap ≥ 0, LOI 0–100 …) | **0** |
| retrosynthesis monomers — parse / sanitise / neutral | **333 / 333 clean (100 %)** |
| cross-property impossibilities (Tg ≥ Tm, Tm > Td) | 5 of 250 (**2 %**) |

The last row is a genuine finding: a small fraction of structures get property pairs that cannot
coexist (a polymer melting above the temperature at which it decomposes). Those cannot be fixed
in the models here, so `chem_review` now **detects and states them** — `cr_tg_above_td` and
`cr_tm_above_td` were added for exactly this — and the audit is repeatable to track the rate.

---

## 3. Copolymers (`copolymers.py`)

Copolymers are more common than blends — ABS, SBR, EVA, SAN, NBR, P(VDF-TrFE) — and more
tractable, because the comonomers are covalently bonded: there is no miscibility question at the
segment level. Three architectures with genuinely different physics:

| architecture | physics | how it is computed |
|---|---|---|
| **alternating** ...ABAB... | the A-B pair **is** a real repeat unit | the unit is **constructed** and handed straight to the property models — a *direct* prediction, not a mixing estimate |
| **random** ...AABABB... | one phase, one Tg | **Fox equation** (its original purpose) + composition-weighted properties |
| **block** ...AAAA-BBBB... | blocks **microphase-separate** when incompatible → two Tgs | blend miscibility screen decides; two Tgs retained |

### Validation
* **Fox on 10 real random copolymers: mean error 7.4 °C** (SBR is the worst at ~23 °C — its own
  homopolymer Tg varies with butadiene microstructure). Includes SAN, P(MMA-co-BA),
  P(VC-co-VAc), P(VDF-co-TrFE), P(S-co-MA), P(VAc-co-MMA).
* **Alternating construction: 10/10** built units are valid, charge-neutral, two-connection-point
  repeat units with **atoms exactly conserved** (heavy atoms = A + B).
* Verified behaviour: the same PS/PBD pair gives **one Tg as a random copolymer (SBR)** and
  **two Tgs as a block copolymer (SBS)** — which is precisely why SBS is a thermoplastic
  elastomer and SBR is a rubber.

### Chemistry the model gets right that a naive average would not
* **Tm is suppressed, not averaged.** Random comonomer incorporation destroys the chain
  regularity crystallisation needs — this is why EVA is flexible where PE is rigid. A weighted
  average of the two melting points would be simply wrong.
* **Td is set by the least stable comonomer**, not averaged.
* **Composition basis**: commercial grades are quoted in **weight** per cent (EVA 28 % VAc,
  SBR 23.5 % styrene), so `basis='weight'` is supported and used by the app. Feeding a weight
  per cent through a mole→weight conversion (the initial bug in the validation script) shifted
  Tg by tens of degrees.

```bash
python validation/validate_copolymers.py
```

---

## Note on xtb and publication

`xtb` is **LGPL-3.0-or-later** (verified from the installed conda-forge package, v6.7.1). Polsen
invokes it as a **separate executable via subprocess** — not linked into the app — which is the
least entangled arrangement under the LGPL, so shipping or deploying Polsen alongside it is fine.
If you ever distribute a *modified* xtb, the LGPL requires you to make those changes available.

If xtb results appear in a paper, cite the method and the program:

* C. Bannwarth, S. Ehlert, S. Grimme, *J. Chem. Theory Comput.* **2019**, 15, 1652–1671
  (GFN2-xTB).
* C. Bannwarth *et al.*, *WIREs Comput. Mol. Sci.* **2021**, 11, e1493 (the xtb program).

The app never requires xtb: the verification panel auto-hides when the binary is absent, so a
deployment without it loses only that panel.

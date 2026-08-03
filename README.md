# Polsen — AI-assisted polymer design

Polsen evolves polymer repeat units toward target properties with a chemistry-aware genetic
algorithm, predicts 16 properties with ML models, proposes a synthesis route, and — unusually
for a tool of this kind — **tells you how much to trust each number**.

Bilingual (Türkçe / English) Streamlit app.

```bash
streamlit run app.py
```

---

## What it does

| | |
|---|---|
| **Design** | Chemistry-aware GA (goal-directed seeding, whole-fragment mutation, structure preservation) evolves repeat units toward your targets. Single-objective or **NSGA-II** multi-objective with a knee-point pick. |
| **Predict** | 16 properties (Tg, Td, Tm, LOI, band gaps, gas permeability, solubility/Hansen, refractive index, dielectric, thermal conductivity, CTE, degradability, recyclability). Refractive index is computed analytically (van Krevelen), the rest by ML models. |
| **Retrosynthesise** | Rule-based backbone disconnection → **real monomer SMILES**, every route verified, each one linked to PubChem. |
| **Verify** | Independent cross-checks of the *specific* result: quantum (GFN2-xTB) and group-contribution (van Krevelen). |
| **Blend / copolymer** | Real materials are blends and copolymers, not neat homopolymers. Screen blend miscibility (Hansen/Flory–Huggins) and estimate properties with standard rules (Fox, Lorentz–Lorenz, Maxwell); for copolymers, random / **alternating** (the A-B unit is built and predicted directly) / block. |
| **Review** | A chemist's-eye pass over each result: flags physically meaningless numbers (a Tm on an amorphous polymer), unstable group pairs, and architecture caveats. |
| **Report** | One-click CSV + full text report with values, units, error bars and provenance. |

## The honesty layer

The distinguishing feature of this project is that every claim is measured, and anything that
didn't validate was left out rather than dressed up.

- **Error bars** — the typical error to expect on a *new* polymer, measured out-of-sample on
  18 canonical handbook polymers, not optimistic in-sample numbers.
- **Per-molecule uncertainty** — for the 9 properties where an uncertainty signal was *proven* to
  predict error (Td, Tm, 3 band gaps, Hansen and gas permeability via bagged companion ensembles;
  Refractive and Solubility via internal signals); marked `＊`. The rest keep the honest global
  average. Log-scaled gas permeability is shown as a **fold** factor (`×/÷ 3.3`), not a
  meaningless symmetric `±`. See [validation/UNCERTAINTY.md](validation/UNCERTAINTY.md).
- **Runner-up candidates** — the lowest-error polymer is not always the easiest to make, so the
  top *N* are listed with their SA scores and properties.
- **Applicability domain** — warns when a polymer sits outside the training distribution, where
  Refractive/EPS in particular revert to the mean.
- **Verified retrosynthesis** — routes are functional-group + atom-conservation checked, and
  flagged ✅ *exact* (the monomer provably re-polymerises to this unit) vs ≈ *approximate*.
  ~92 % of GA outputs get a route; **0 % get a wrong one**. A "no route" is an honest
  novelty flag, not a failure.

## Repository layout

```
app.py              the app (CONFIG flags at the top)
smart_ga.py           chemistry-aware GA operators
retro.py              rule-based retrosynthesis engine
van_krevelen.py       group-contribution property estimator (independent check)
blends.py             blend miscibility + mixing rules
copolymers.py         random / alternating / block copolymers
chem_review.py        chemist-facing validity notes
xtb_tools.py          GFN2-xTB quantum cross-check (optional, local only)
lang_dict.py          TR/EN strings
*.joblib              the 16 property models
bagged_model.py       BaggedEnsemble container
models_bagged/        bagged companion ensembles (uncertainty only, not predictions)
tests/                regression suites — run after any change
validation/           evidence: how good the models are, and how we know
md/                   offline molecular-dynamics pipeline (Tg / CTE / density)
Dockerfile            container build (installs xtb); .dockerignore trims the image
versions/             snapshots — v4 = core app fully validated (pre-blend, see its
                      VERSION.md); v3_son = before the bagged-ensemble retrain
```

## Configuration

Feature flags at the top of `app.py`:

| flag | effect |
|---|---|
| `SHOW_RELIABILITY` | reliability badges, `±` error bars, and their export columns |
| `SHOW_APPLICABILITY` | out-of-domain warning banner |
| `SHOW_MANUAL_ANALYSIS` | "analyse my own SMILES" panel |
| `SHOW_PARETO_TABLE` | Pareto front table/plot (NSGA-II) |
| `USE_T5_RETRO` | the heavy T5 retro model (off = rule engine only, recommended) |
| `ENABLE_XTB` | quantum verification panel (auto-hides where xtb isn't installed) |
| `USE_BAGGED_UNCERTAINTY` | molecule-specific error bars from the bagged companions (~49 MB, lazy) |
| `TOP_N_CANDIDATES` | how many runner-up polymers to list (0 hides the section) |
| `SHOW_BLENDS` | the Blend tab (miscibility screen + blend property estimates) |
| `SHOW_CHEM_REVIEW` | chemist-facing validity notes on each result |

Sidebar options worth knowing: **random seed** (reproducible runs), **start from your own
polymer** (improve an existing structure instead of starting fresh), **advanced GA parameters**,
and **"stay in the model's confidence zone"** — off by default, because it trades novelty for
prediction trustworthiness.


## Deploying (Railway / any Docker host)

The repo root has a `Dockerfile`; Railway picks it up automatically. It installs the **official
xtb Linux binary** (27 MB, pinned to v6.7.1 and checksum-verified) so the quantum verification
panel works in production. Remove that block for a slimmer image — the panel just auto-hides.

```bash
railway up          # or: docker build -t polsen . && docker run -p 8080:8080 polsen
```

`.dockerignore` keeps the image lean by dropping `versions/`, `models_bagged/unused/`, `md/`,
`tests/` and the validation scripts — but **keeps `validation/uncertainty_calibration.json`**,
which the app loads at runtime for the per-molecule error bars. Build context ≈ 99 MB.

`USE_T5_RETRO` is **off** and `torch` / `transformers` / `google-generativeai` are commented out
of `requirements.txt`, so the image carries no multi-GB ML runtime: the T5 was trained on
synthetic data and does not generalise, while the rule engine alone reaches ~92 % verified /
0 % wrong. Verified to boot in ~4 s with torch genuinely unavailable. Turn the flag back on only
if you also uncomment those dependencies.

`.gitignore` keeps the repository at **99 MB** (the working folder is 693 MB) by excluding
`versions/`, `models_bagged/unused/` and the unrelated `worldmapper_quiz/`; no file exceeds
GitHub's 50 MB warning threshold. The models themselves *are* committed — the app needs them.

## Tests

```bash
python tests/test_retro.py          # retrosynthesis rules (fast)
python tests/test_translations.py   # TR/EN coverage incl. retro route names
python tests/test_app_results.py    # app results path (loads models)
```

Run them after touching `retro.py`, `smart_ga.py`, `app.py`, or `lang_dict.py`.
⚠️ Never write `x, _ = f()` at module level in `app.py` — `_` is the translation function and
shadowing it crashes the app. `test_app_results.py` guards against this.

## Validation evidence

| document | what it establishes |
|---|---|
| [GOLDEN_SET_VALIDATION.md](validation/GOLDEN_SET_VALIDATION.md) | out-of-sample accuracy per property on canonical polymers |
| [XTB_VALIDATION.md](validation/XTB_VALIDATION.md) | band gap / refractive / dielectric confirmed by quantum chemistry |
| [VAN_KREVELEN.md](validation/VAN_KREVELEN.md) | independent group-contribution estimates (and what it can't do) |
| [UNCERTAINTY.md](validation/UNCERTAINTY.md) | which uncertainty signals predict error — and which don't |
| [APP_TEST_REPORT.md](validation/APP_TEST_REPORT.md) | end-to-end test on 6 real design missions, with the chemistry judged by hand |
| [BLENDS_AND_CHEM_REVIEW.md](validation/BLENDS_AND_CHEM_REVIEW.md) | blend miscibility 23/28, Fox Tg ±4 °C, crystallinity 10/10, plus the app-wide chemistry audit |
| [md/README.md](md/README.md) | MD pipeline scope: Tg/CTE/density are tractable; Tm/Td/κ/permeability are not |

## Optional extras (local only, never required)

```bash
conda install -c conda-forge xtb                       # quantum verification panel
conda create -n polymd -c conda-forge python=3.11 rdkit numpy openmm \
    openff-toolkit-base openff-interchange             # MD pipeline
```

Neither is a deployment dependency: the app hides the xtb panel when it isn't present, and never
calls MD.

---

*Predictions are ML estimates, not measurements. Use the error bars, the applicability-domain
warning, and the verification panel before trusting a number.*

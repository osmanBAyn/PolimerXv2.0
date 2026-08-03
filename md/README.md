# MD property estimation (`md/polymer_md.py`)

Molecular dynamics is the rigorous route for the bulk properties no cheap method verifies well.
But be clear-eyed about what "use MD" actually buys — even MD does not make every property easy.

## What this pipeline delivers (OpenMM, tractable, tested here)

| Property | Method | Runtime (this GPU) | Status |
|---|---|---|---|
| **Density** | NPT equilibration | ~minutes | ✅ works |
| **Tg** | specific-volume vs temperature scan, knee | ~1–3 h | ✅ works |
| **CTE** | slope of V(T) above Tg (same scan) | free with Tg | ✅ works |
| **Solubility** δ | cohesive energy density | ~10 min | 🟡 v2 |

`Tg` and `CTE` come from **one** temperature scan, so they are the main new things MD adds that
van Krevelen and xtb cannot verify per-polymer.

## What MD still does NOT make easy (honest scope)

These each need a specialized method or force field — they are individual research efforts, not a
switch on this pipeline:

- **Tm** (crystalline melting) — needs crystal building + slow nucleation/melting; a general
  amorphous run does not give it.
- **Td** (thermal decomposition) — needs a *reactive* force field (ReaxFF). OpenFF/GAFF cannot
  break bonds, so decomposition is out of scope here.
- **Thermal conductivity** — Green–Kubo (long equilibrium heat-flux autocorrelation) or NEMD;
  hours–days and finicky to converge.
- **Gas permeability** — solubility (Widom insertion) × diffusivity (long MSD runs); hours–days
  per gas.

So MD's realistic contribution to *this* project is a rigorous **Tg / CTE / density** check on a
shortlisted candidate — not a universal oracle for all 13 non-quantum properties.

## Accuracy caveats

- **Charges: Gasteiger.** AmberTools has no Windows conda build, so AM1-BCC is unavailable here
  and the pipeline falls back to RDKit Gasteiger charges — fine for density/Tg screening, but for
  publication-grade numbers run on Linux with `ambertools` (AM1-BCC) or install `openff-nagl`.
- **Small & short by default** so a run *finishes*: a few short oligomer chains, sub-nanosecond
  production. Increase `--units`, `--chains`, `--ns` for convergence; a real Tg wants larger cells
  and longer runs. Treat defaults as screening.
- **Amorphous only** — no crystallinity, so semicrystalline Tm/behavior is not represented.

## Status: tested & working

Installed and verified on this machine (Windows, RTX 5070 Ti via **OpenCL**; no CUDA platform in
the conda OpenMM build, OpenCL still runs on the GPU). Polyethylene density ran end-to-end and
plateaus at **0.78–0.79 g/cm³** (0.1 → 0.4 ns equilibration) vs ~0.85 for amorphous PE — a ~7 %
underestimate that is the expected *screening* gap: short 12-mer chains (many chain ends lower
density), Gasteiger charges, and the trimmed 0.8 nm cutoff. For publication numbers use longer
chains, AM1-BCC charges (Linux/`ambertools`), the full cutoff, and longer runs. `packmol` and
`ambertools` have no conda win-64 build, so the pipeline uses its **own grid packer** (no packmol)
and **Gasteiger charges** (no AM1-BCC).

## Setup (dedicated env — keeps the app's `polsen` env untouched)

```bash
conda create -n polymd -c conda-forge --solver=libmamba -y \
  python=3.11 rdkit numpy openmm openff-toolkit-base openff-interchange
conda run -n polymd python md/polymer_md.py "*CC(c1ccccc1)*" --mode density
conda run -n polymd python md/polymer_md.py "*CC(c1ccccc1)*" --mode tg --units 12
```

(`openff-toolkit-base`, not `openff-toolkit`, avoids the ambertools dependency that fails on
Windows. `openmmforcefields`/`packmol` are intentionally omitted — not needed and not on win-64.)

## Why it is offline-only

A density point is minutes; a Tg scan is hours. It cannot run per-GA-result, and it cannot run on
Streamlit Cloud (no GPU, no MD engine). Run it from the shell on a final candidate.

For a fully-automated, HPC-grade version of all of the above (with QM charges and validated
protocols), **RadonPy** is the reference tool — it wraps LAMMPS and is best run on Linux/HPC.

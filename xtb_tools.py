"""
xtb_tools.py — optional local quantum cross-check for Polsen (GFN2-xTB).

Pure functions: no streamlit, no appv3 import. Used lazily by app.py (Manual Polymer
Analysis) and by validation/xtb_crosscheck.py. `xtb` is an OFFLINE convenience, NOT a
deployment dependency — when the executable is absent, find_xtb() returns None and the
app hides the feature. Install locally with:  conda install -c conda-forge xtb

For one polymer repeat unit ("*...*" SMILES) crosscheck_one() returns, from GFN2-xTB:
  gap_inf   : HOMO-LUMO gap extrapolated to the infinite chain (eV)  -> compare to BandgapChain
  gap_n1    : gap of the single H-capped repeat unit (eV)
  alpha     : molecular electronic polarizability (atomic units)
  vol       : molecular volume (A^3, RDKit)
  alpha_vol : alpha / vol  -> electronic polarizability density; ranks with refractive index
              (Lorentz-Lorenz) and dielectric constant (Clausius-Mossotti)

Rank/agreement is the point: GFN2 is semi-empirical, so absolute gaps sit systematically
off the DFT the models were trained on. See validation/XTB_VALIDATION.md.
"""
import os, re, shutil, subprocess, tempfile
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# Known local install (conda-forge). Env var XTB_EXE and PATH take precedence.
_KNOWN_PATHS = [
    r"C:\Users\osbar\miniconda3\envs\polsen\Library\bin\xtb.exe",
]


def find_xtb():
    """Locate an xtb executable, or return None if not installed."""
    p = os.environ.get("XTB_EXE")
    if p and os.path.exists(p):
        return p
    w = shutil.which("xtb")
    if w:
        return w
    for k in _KNOWN_PATHS:
        if os.path.exists(k):
            return k
    return None


def _xyz(mol):
    """3D-embed + MMFF-optimize; return (xyz_string, mol_with_Hs) or (None, None)."""
    mh = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mh, randomSeed=42) != 0:
        if AllChem.EmbedMolecule(mh, randomSeed=1, useRandomCoords=True) != 0:
            return None, None
    try:
        AllChem.MMFFOptimizeMolecule(mh, maxIters=500)
    except Exception:
        pass
    conf = mh.GetConformer()
    lines = [str(mh.GetNumAtoms()), ""]
    for at in mh.GetAtoms():
        pos = conf.GetAtomPosition(at.GetIdx())
        lines.append(f"{at.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
    return "\n".join(lines), mh


def _sp(xyz, xexe, timeout=180):
    """One GFN2 single point -> (gap_eV, alpha_au). Either may be None if unparsed."""
    with tempfile.TemporaryDirectory() as td:
        open(os.path.join(td, "m.xyz"), "w").write(xyz)
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        # latin-1/replace: xtb's banner has box-drawing/Greek glyphs that crash Windows cp1254.
        out = subprocess.run([xexe, "m.xyz", "--gfn", "2", "--sp"], cwd=td,
                             capture_output=True, text=True, timeout=timeout,
                             env=env, encoding="latin-1", errors="replace").stdout
    g = re.search(r"HOMO-LUMO GAP\s+([-\d.]+)\s+eV", out)
    # "Mol. a(0) /au : value"; the Greek letter is mangled by latin-1, so anchor on "/au :".
    a = re.search(r"Mol\.[^:\n]*?/au\s*:\s*([-\d.]+)", out)
    return (float(g.group(1)) if g else None,
            float(a.group(1)) if a else None)


def build_oligomer(smi, n):
    """Join n copies of a '*...*' repeat unit head-to-tail, cap the two chain ends with H."""
    unit = Chem.MolFromSmiles(smi)
    if unit is None:
        return None
    stars = [a.GetIdx() for a in unit.GetAtoms() if a.GetAtomicNum() == 0]
    if len(stars) != 2:
        return None
    nb = []
    for s in stars:
        ns = unit.GetAtomWithIdx(s).GetNeighbors()
        if not ns:
            return None
        nb.append(ns[0].GetIdx())
    (head_star, tail_star), (head_nb, tail_nb) = stars, (nb[0], nb[1])
    na = unit.GetNumAtoms()
    combo = Chem.RWMol(unit)
    for _ in range(n - 1):
        combo.InsertMol(unit)
    rm = []
    for i in range(n):
        off = i * na
        if i < n - 1:
            combo.AddBond(off + tail_nb, (i + 1) * na + head_nb, Chem.BondType.SINGLE)
            rm += [off + tail_star, (i + 1) * na + head_star]
    rm += [head_star, (n - 1) * na + tail_star]      # remove the two terminal stars -> H caps
    for idx in sorted(set(rm), reverse=True):
        combo.RemoveAtom(idx)
    m = combo.GetMol()
    try:
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


# ---------------------------------------------------------------------------------------
# Per-polymer VERIFICATION calibrations.
# GFN2-xTB has a large *systematic* offset from the DFT the models were trained on, and
# alpha/V is only proportional to the Lorentz-Lorenz / Clausius-Mossotti functions. To turn
# a raw xtb number into an estimate directly comparable to a model prediction, we remove that
# systematic part with a linear map fitted on the validation polymers (see validation/), and
# report the *residual* scatter as the honest per-polymer uncertainty. Provenance:
#   band gap : DFT ~= 0.867*xtb_inf + 2.275   (22 polymers; residual RMSE 0.86 eV)
#   ref.index: LL(n) ~= 0.320*(alpha/V) + 0.172 (22 polymers; residual RMSE 0.05 in n)
GAP_CAL = (0.867, 2.275); GAP_UNC = 0.9      # eV, ~1 sigma -> a genuine absolute check
# Refractive index from alpha/V is only RANKING-grade: RDKit molecular volume is not the
# packed-polymer density, and the packing fraction varies per polymer (e.g. loosely-packed
# aromatics like polystyrene read ~0.2 too high). So the RI/EPS checks are deliberately SOFT
# consistency indicators, not tight verifications. Band gap is the one tight absolute check.
RI_CAL = (0.320, 0.172);  RI_UNC = 0.10      # widened to reflect per-polymer (out-of-dist) scatter


def _status(diff, unc):
    """Agreement verdict from |model - estimate| vs the estimate's uncertainty."""
    if diff <= 2 * unc:
        return 'ok'          # within ~2 sigma -> consistent
    if diff <= 4 * unc:
        return 'warn'        # borderline
    return 'bad'             # model and independent estimate genuinely diverge


def verify_predictions(smi, model_preds, xexe=None, xr=None):
    """
    Independently verify ONE polymer's model predictions with GFN2-xTB.
    Returns {'raw': crosscheck dict, 'checks': [ {prop, model, estimate, unc, unit,
    status, kind, note}, ... ], 'error': str|None}. Only the electronic-structure
    properties (band gap, refractive index, dielectric) are xtb-verifiable; the panel
    simply omits the rest. `model_preds` is the app's preds dict for this polymer.
    """
    if xr is None:
        xr = crosscheck_one(smi, xexe)
    out = {'raw': xr, 'checks': [], 'error': None}
    if xr.get('gap_inf') is None and xr.get('alpha_vol') is None:
        out['error'] = xr.get('error') or 'no usable xtb result'
        return out

    # --- band gap (chain): calibrated absolute comparison ---
    if xr.get('gap_inf') is not None and model_preds.get('BandgapChain') is not None:
        a, b = GAP_CAL
        est = a * xr['gap_inf'] + b
        mv = float(model_preds['BandgapChain'])
        out['checks'].append({'prop': 'BandgapChain', 'model': mv, 'estimate': est,
                              'unc': GAP_UNC, 'unit': 'eV', 'status': _status(abs(mv - est), GAP_UNC),
                              'kind': 'estimate', 'method': 'xtb (GFN2)', 'note': ''})

    # --- optical dielectric floor from electronic polarizability density ---
    # (van Krevelen handles the refractive-index card; here we only derive n^2 for the EPS floor.)
    eps_inf = None
    if xr.get('alpha_vol') is not None:
        a, b = RI_CAL
        ll = min(max(a * xr['alpha_vol'] + b, 0.0), 0.95)
        eps_inf = (1 + 2 * ll) / (1 - ll)        # optical/electronic dielectric = n^2

    # --- dielectric: physical LOWER-BOUND sanity (static eps must exceed optical eps_inf=n^2) ---
    if eps_inf is not None and model_preds.get('EPS') is not None:
        mv = float(model_preds['EPS'])
        # electronic part only; static eps adds a dipolar term, so a valid model sits >= eps_inf.
        # Only flag a GROSS violation (> 1.0 below the noisy floor) to avoid calibration false alarms.
        status = 'warn' if mv < eps_inf - 1.0 else 'ok'
        out['checks'].append({'prop': 'EPS', 'model': mv, 'estimate': eps_inf, 'unc': None,
                              'unit': '', 'status': status, 'kind': 'lower_bound', 'method': 'xtb (GFN2)',
                              'note': 'optical floor eps_inf = n^2; static eps should be >= this'})
    return out


def crosscheck_one(smi, xexe=None, ns=(1, 2, 3), timeout=180, max_heavy=90):
    """
    Quantum cross-check one repeat-unit SMILES. Returns a dict (see module docstring).
    Runs one xtb single point per oligomer length in `ns`; n=1 also yields alpha/vol.
    """
    res = {'gap_inf': None, 'gap_n1': None, 'alpha': None, 'vol': None,
           'alpha_vol': None, 'n_points': 0, 'error': None}
    if xexe is None:
        xexe = find_xtb()
    if xexe is None:
        res['error'] = 'xtb executable not found'
        return res
    if not smi or str(smi).count('*') != 2:
        res['error'] = 'need a repeat-unit SMILES with exactly two "*" endpoints'
        return res
    try:
        xs, ys = [], []
        for n in ns:
            oli = build_oligomer(smi, n)
            if oli is None or oli.GetNumHeavyAtoms() > max_heavy:
                continue
            xyz, mh = _xyz(oli)
            if xyz is None:
                continue
            gap, alpha = _sp(xyz, xexe, timeout)
            if gap is not None:
                xs.append(1.0 / n)
                ys.append(gap)
            if n == 1:
                res['gap_n1'] = gap
                res['alpha'] = alpha
                try:
                    res['vol'] = AllChem.ComputeMolVolume(mh)
                except Exception:
                    res['vol'] = None
        res['n_points'] = len(ys)
        if len(ys) >= 2:
            slope, intercept = np.polyfit(xs, ys, 1)   # gap = slope*(1/n) + intercept
            res['gap_inf'] = float(intercept)           # 1/n -> 0 : infinite chain
        elif len(ys) == 1:
            res['gap_inf'] = ys[0]
        if res['alpha'] and res['vol']:
            res['alpha_vol'] = res['alpha'] / res['vol']
        if res['gap_inf'] is None and res['alpha'] is None:
            res['error'] = 'xtb ran but produced no usable gap/polarizability (embedding failed?)'
    except Exception as e:
        res['error'] = str(e)[:200]
    return res

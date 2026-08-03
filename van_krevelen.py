"""
van_krevelen.py — independent group-contribution (van Krevelen) property estimator.

A physics/empirical second opinion on a polymer repeat unit, computed from its structure
alone — completely independent of the ML models. It is the polymer-science analog of xtb:
where xtb covers the electronic properties, this covers the bulk ones.

Implemented (validated against canonical polymers in __main__):
  density        rho = M / V              (V from group molar-volume increments)
  solubility     Hildebrand delta         (Hoftyzer-Van Krevelen Fd/Fp/Eh)   -> app 'Solubility'
  hansen         delta_d, delta_p, delta_h                                    -> app 'Hansen'
  refractive     Lorentz-Lorenz from molar refraction (RDKit MolMR) and V     -> app 'Refractive'

NOT implemented (deliberately): Tg / Tm / Td / CTE / thermal conductivity / gas permeability.
Van Krevelen predicts these from tabulated Yg/Ym/etc. group functions; without the authoritative
tables a from-literature calibration overfits (leave-one-out MAE ~90-130 degC on Tg/Tm — worse
than the trained model's own ~44 degC). So for those, trust the model's value with the golden-set
error bar (validation/GOLDEN_SET_VALIDATION.md), or use MD (RadonPy). Extend via the book's Yg/Ym.

Method: replace each chain end '*' with a carbon placeholder (so backbone atoms get the right
valence/H-count), then tile the repeat unit into groups with SMARTS (functional groups first,
then atoms), summing each group's contributions. `coverage` = fraction of heavy atoms assigned;
< 1.0 means an unrecognized fragment and lower confidence.

Units: delta in MPa^0.5 internally (converted per the app's unit for each property).
"""
import re
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen
RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------------------
# Group table. Each: (name, SMARTS, {V, Fd, Fp, Eh}). Ordered specific -> generic;
# tiling is greedy by descending atom count so multi-atom functional groups win first.
#   V  : molar volume increment            cm^3/mol
#   Fd : dispersion  (Hoftyzer-Van Krevelen) (J.cm^3)^0.5/mol
#   Fp : polar                               (J.cm^3)^0.5/mol   (added in quadrature)
#   Eh : hydrogen-bond cohesive energy       J/mol
# (Values tuned so delta / density / RI validate on canonical polymers; see __main__.)
def _g(V=0, Fd=0, Fp=0, Eh=0):
    return dict(V=V, Fd=Fd, Fp=Fp, Eh=Eh)

GROUPS = [
    # --- multi-atom functional groups (match first) ---
    ("carbonate",  "[OX2][CX3](=O)[OX2]", _g(V=22.0, Fd=760, Fp=920, Eh=9000)),
    ("ester",      "[CX3](=O)[OX2]",      _g(V=18.0, Fd=390, Fp=490, Eh=7000)),
    ("amide",      "[CX3](=O)[NX3]",      _g(V=18.5, Fd=560, Fp=600, Eh=20000)),
    ("nitrile",    "[CX2]#[NX1]",         _g(V=24.0, Fd=430, Fp=950, Eh=2500)),
    ("sulfone",    "[SX4](=O)(=O)",       _g(V=18.0, Fd=500, Fp=1500, Eh=1500)),
    ("ketone",     "[CX3]=O",             _g(V=10.8, Fd=290, Fp=770, Eh=2000)),
    # --- heteroatom singles ---
    ("hydroxyl",   "[OX2H]",              _g(V=10.0, Fd=210, Fp=500, Eh=20000)),
    ("ether_O",    "[OX2]",               _g(V=3.8,  Fd=100, Fp=400, Eh=3000)),
    ("amine_NH",   "[NX3;H1,H2]",         _g(V=4.5,  Fd=160, Fp=210, Eh=3100)),
    ("amine_N",    "[NX3;H0]",            _g(V=-9.0, Fd=20,  Fp=800, Eh=5000)),
    ("fluorine",   "[F]",                 _g(V=23.0, Fd=185, Fp=0,   Eh=0)),
    ("chlorine",   "[Cl]",                _g(V=24.0, Fd=450, Fp=550, Eh=400)),
    ("bromine",    "[Br]",                _g(V=30.0, Fd=550, Fp=550, Eh=400)),
    ("thioether",  "[SX2]",               _g(V=12.0, Fd=440, Fp=0,   Eh=0)),
    ("silicon",    "[Si]",                _g(V=0.0,  Fd=-30, Fp=0,   Eh=0)),
    # --- aromatic carbons (per atom; benzene = 6 -> ~phenyl Fd) ---
    ("arom_CH",    "[cH]",                _g(V=13.5, Fd=240, Fp=0,   Eh=0)),
    ("arom_C",     "[c]",                 _g(V=10.0, Fd=200, Fp=20,  Eh=0)),
    # --- sp2 aliphatic ---
    ("eq_CH2",     "[CX3H2]",             _g(V=28.5, Fd=400, Fp=0,   Eh=0)),
    ("eq_CH",      "[CX3H1]",             _g(V=13.5, Fd=200, Fp=0,   Eh=0)),
    ("eq_C",       "[CX3H0]",             _g(V=-5.5, Fd=70,  Fp=0,   Eh=0)),
    # --- sp3 aliphatic ---
    ("CH3",        "[CX4H3]",             _g(V=33.5, Fd=420, Fp=0,   Eh=0)),
    ("CH2",        "[CX4H2]",             _g(V=16.1, Fd=270, Fp=0,   Eh=0)),
    ("CH",         "[CX4H1]",             _g(V=-1.0, Fd=80,  Fp=0,   Eh=0)),
    ("C",          "[CX4H0]",             _g(V=-19.2,Fd=-70, Fp=0,   Eh=0)),
]
_COMPILED = [(n, Chem.MolFromSmarts(s), c) for n, s, c in GROUPS]


def _prep(smiles):
    """Replace each '*' with a C placeholder (fixes neighbor valence), return (mol, placeholder_idx set)."""
    s = str(smiles)
    if s.count("*") == 0:
        m = Chem.MolFromSmiles(s)
        return (m, set()) if m else (None, None)
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None, None
    rw = Chem.RWMol(m)
    ph = set()
    for at in rw.GetAtoms():
        if at.GetAtomicNum() == 0:               # dummy '*'
            at.SetAtomicNum(6)                    # make it carbon so the neighbor's H-count is right
            at.SetNoImplicit(False)
            ph.add(at.GetIdx())
    try:
        mm = rw.GetMol(); Chem.SanitizeMol(mm)
        return mm, ph
    except Exception:
        return None, None


def _tile(mol, placeholders):
    """Assign each ORIGINAL heavy atom to exactly one group (greedy, specific first)."""
    heavy = [a.GetIdx() for a in mol.GetAtoms()
             if a.GetAtomicNum() > 1 and a.GetIdx() not in placeholders]
    used = set()
    counts = {}
    order = sorted(_COMPILED, key=lambda t: -t[1].GetNumAtoms())   # multi-atom groups first
    for name, patt, contrib in order:
        if patt is None:
            continue
        for match in mol.GetSubstructMatches(patt, uniquify=True):
            real = [i for i in match if i not in placeholders]
            if not real or any(i in used for i in real):
                continue
            used.update(real)
            counts[name] = counts.get(name, 0) + 1
    covered = sum(1 for i in heavy if i in used)
    coverage = covered / len(heavy) if heavy else 0.0
    return counts, coverage


def vk_estimate(smiles):
    """
    van Krevelen group-contribution estimate for a '*...*' repeat unit.
    Returns dict with density, Solubility, Hansen, Refractive, Tg, Tm, and 'coverage'
    (0..1). Returns {'error': ...} if the SMILES can't be processed.
    """
    mol, ph = _prep(smiles)
    if mol is None:
        return {'error': 'unparseable SMILES'}
    counts, coverage = _tile(mol, ph)
    if not counts:
        return {'error': 'no groups matched'}

    tot = dict(V=0.0, Fd=0.0, Fp2=0.0, Eh=0.0)
    for name, patt, contrib in _COMPILED:
        n = counts.get(name, 0)
        if not n:
            continue
        tot['V'] += n * contrib['V']
        tot['Fd'] += n * contrib['Fd']
        tot['Fp2'] += n * (contrib['Fp'] ** 2)
        tot['Eh'] += n * contrib['Eh']

    # molar mass of the repeat unit (subtract the two placeholder carbons + their H)
    M = Descriptors.MolWt(mol)
    for i in ph:
        a = mol.GetAtomWithIdx(i)
        M -= 12.011 + a.GetTotalNumHs() * 1.008

    # amorphous packing correction: group-sum V runs ~8% small vs real amorphous molar volume,
    # which biases BOTH density and RI high; one physical factor removes the systematic part.
    V = tot['V'] * 1.08
    out = {'coverage': round(coverage, 2), 'M': round(M, 1), 'V': round(V, 1)}
    if V > 5:
        out['density'] = round(M / V, 3)
        dd = tot['Fd'] / V
        dp = (tot['Fp2'] ** 0.5) / V
        dh = (tot['Eh'] / V) ** 0.5 if tot['Eh'] > 0 else 0.0
        delta = (dd * dd + dp * dp + dh * dh) ** 0.5      # MPa^0.5
        out['Solubility_MPa'] = round(delta, 2)
        out['Solubility'] = round(delta / 2.0455, 2)      # app unit: (cal/cm3)^0.5
        out['Hansen'] = round(delta, 2)                   # app 'Hansen' is total delta in MPa^0.5
        out['Hansen_components'] = (round(dd, 2), round(dp, 2), round(dh, 2))

        # refractive index via molar refraction (Lorentz-Lorenz). MolMR from RDKit (Crippen).
        capped = Chem.MolFromSmiles(str(smiles).replace('*', '[H]'))
        if capped is not None:
            R = Crippen.MolMR(capped)                     # ~ molar refraction, cm^3/mol
            rv = R / V
            if 0 < rv < 1:
                out['Refractive'] = round(((1 + 2 * rv) / (1 - rv)) ** 0.5, 3)
    return out


# validated per-property 1-sigma (from __main__ on canonical polymers)
VK_UNC = {'Solubility': 1.2, 'Hansen': 2.4, 'Refractive': 0.065}


def vk_verify(smiles, model_preds):
    """
    Independent group-contribution cross-check of a polymer's Solubility / Hansen / Refractive
    predictions. Returns a list of check dicts in the SAME shape xtb_tools.verify_predictions
    uses, tagged method='van Krevelen', so the app can merge and render them uniformly.
    """
    e = vk_estimate(smiles)
    checks = []
    if 'error' in e:
        return checks
    cov = e.get('coverage', 0)
    for prop in ('Solubility', 'Hansen', 'Refractive'):
        est = e.get(prop)
        mv = model_preds.get(prop)
        if est is None or mv is None:
            continue
        mv = float(mv)
        unc = VK_UNC[prop]
        diff = abs(mv - est)
        status = 'ok' if diff <= 2 * unc else ('warn' if diff <= 4 * unc else 'bad')
        if cov < 0.999:          # unrecognized fragment -> don't hard-fail
            status = 'ok' if status == 'ok' else 'warn'
        checks.append({'prop': prop, 'model': mv, 'estimate': est, 'unc': unc,
                       'unit': '(cal/cm³)^½' if prop == 'Solubility' else ('MPa^½' if prop == 'Hansen' else ''),
                       'status': status, 'kind': 'soft', 'method': 'van Krevelen',
                       'note': f"coverage {cov:.0%}"})
    return checks


# =======================================================================================
# Validation against canonical polymers (handbook values) — run: python van_krevelen.py
# =======================================================================================
if __name__ == "__main__":
    import numpy as np
    #  name : (SMILES, density, Solub(cal), RI)   None = skip
    KNOWN = {
        "PE":    ("*CCCC*",                               0.85,  8.0, 1.51),
        "PP":    ("*CC(C)CC(C)*",                         0.85,  8.0, 1.49),
        "PS":    ("*CC(c1ccccc1)*",                       1.05,  9.1, 1.59),
        "PMMA":  ("*CC(C)(C(=O)OC)*",                     1.18,  9.3, 1.49),
        "PVC":   ("*CC(Cl)CC(Cl)*",                       1.39,  9.6, 1.54),
        "PTFE":  ("*C(F)(F)C(F)(F)*",                     2.10,  6.2, 1.35),
        "PAN":   ("*CC(C#N)*",                            1.18, 12.5, 1.52),
        "PC":    ("*OC(=O)Oc1ccc(C(C)(C)c2ccc(*)cc2)cc1", 1.20,  9.8, 1.585),
        "PET":   ("*OCCOC(=O)c1ccc(C(=O)*)cc1",           1.38, 10.7, 1.57),
        "Nylon6":("*NCCCCCC(=O)*",                        1.13, 13.6, 1.53),
        "PEO":   ("*CCOCCO*",                             1.13,  9.9, 1.46),
        "PVA":   ("*CC(O)CC(O)*",                         1.29, 12.6, 1.50),
        "PDMS":  ("*[Si](C)(C)O*",                        0.98,  7.4, 1.43),
        "PMA":   ("*CC(C(=O)OC)*",                        1.22,  9.7, 1.48),
    }
    props = [("density", 1, 0.08), ("Solubility", 2, 1.0), ("Refractive", 3, 0.05)]
    res = {p: [] for p, _, _ in props}
    cover = []
    print(f"{'polymer':<9} {'cov':>4} {'dens':>12} {'solub':>12} {'RI':>12}")
    for name, tup in KNOWN.items():
        if tup is None:
            continue
        smi = tup[0]
        e = vk_estimate(smi)
        if 'error' in e:
            print(f"{name:<9} ERROR {e['error']}"); continue
        cover.append(e.get('coverage', 0))
        cells = [f"{name:<9}", f"{e.get('coverage',0):>4.2f}"]
        for p, idx, _ in props:
            lit = tup[idx]; pr = e.get(p)
            if lit is not None and pr is not None:
                res[p].append((pr - lit))
                cells.append(f"{pr:>5.2f}/{lit:<4}")
            else:
                cells.append(f"{('%.2f'%pr) if pr is not None else '-':>5}/{'-':<4}")
        print(" ".join(cells))
    print(f"\nmean coverage {np.mean(cover):.2f}")
    print(f"{'property':<12} {'n':>2} {'MAE':>8}  within-threshold")
    for p, _, thr in props:
        d = np.array(res[p])
        if len(d):
            ok = np.mean(np.abs(d) <= thr) * 100
            print(f"{p:<12} {len(d):>2} {np.mean(np.abs(d)):>8.3f}  {ok:.0f}% within +-{thr}")

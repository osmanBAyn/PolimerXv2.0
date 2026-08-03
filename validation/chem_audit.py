"""
chem_audit.py — app-wide chemical/physical validity audit.

Checks every predicted property against (a) hard physical bounds and (b) cross-property
relations that must hold for any real polymer, on both canonical polymers and freshly
generated GA structures. Anything that fails is a number the app would show a chemist.

    python validation/chem_audit.py [n_ga]

Bounds are deliberately generous — they flag the physically IMPOSSIBLE, not the merely unusual.
"""
import sys, types, warnings, os, random
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)

class _Ctx:
    def __enter__(s): return s
    def __exit__(s, *a): return False
    def __getattr__(s, n): return lambda *a, **k: _Ctx()
class _SS(dict):
    def __getattr__(s, n):
        try: return s[n]
        except KeyError: raise AttributeError(n)
    def __setattr__(s, n, v): s[n] = v
class _St(types.ModuleType):
    def __getattr__(s, n):
        if n.startswith("__") and n.endswith("__"): raise AttributeError(n)
        return lambda *a, **k: _Ctx()
st = _St("streamlit"); st.session_state = _SS()
def _c(*a, **k):
    if a and callable(a[0]): return a[0]
    return lambda f: f
st.cache_data = _c; st.cache_resource = _c
st.columns = lambda spec, *a, **k: [_Ctx() for _ in range(spec if isinstance(spec, int) else len(spec))]
st.tabs = lambda l, *a, **k: [_Ctx() for _ in l]
st.selectbox = lambda l, o=(), *a, **k: (list(o)[k.get("index", 0)] if o else None); st.radio = st.selectbox
st.checkbox = lambda *a, **k: bool(k.get("value", False)); st.button = lambda *a, **k: False
st.slider = lambda l, mn=0, mx=1, val=None, *a, **k: (val if val is not None else mn)
st.number_input = lambda *a, **k: k.get("value", 0.0); st.text_input = lambda *a, **k: ""
st.stop = lambda *a, **k: None; st.set_page_config = lambda *a, **k: None; st.sidebar = st; st.__path__ = []
_cc = _St("streamlit.components"); _cc.__path__ = []; _v = _St("streamlit.components.v1"); _cc.v1 = _v; st.components = _cc
sys.modules["streamlit"] = st; sys.modules["streamlit.components"] = _cc; sys.modules["streamlit.components.v1"] = _v
import datasets as d
_SEEDS = ["*CC(c1ccccc1)*", "*CCO*", "*OCCOC(=O)c1ccc(C(=O)*)cc1", "*NCCCCCC(=O)*",
          "*[Si](C)(C)O*", "*OC(=O)Oc1ccc(C(C)(C)c2ccc(*)cc2)cc1", "*CC(C)(C(=O)OC)*",
          "*Oc1ccc(cc1)S(=O)(=O)c1ccc(*)cc1", "*OC(C)C(=O)*", "*C(F)(F)C(F)(F)*",
          "*CC=CC*", "*CC(C#N)*", "*CC(Cl)*", "*COC*", "*OCCCCCC(=O)*"]
d.load_dataset = lambda *a, **k: type("D", (), {"to_pandas": lambda s: pd.DataFrame({"p_smiles": _SEEDS})})()
sys.path.insert(0, PROJ)
import importlib.util
_spec = importlib.util.spec_from_file_location("appv3", os.path.join(PROJ, "app.py"))
A = importlib.util.module_from_spec(_spec); sys.modules["appv3"] = A; _spec.loader.exec_module(A)
import smart_ga as sga, retro, chem_review as cr

# ---- hard physical bounds (flag the IMPOSSIBLE, not the unusual) --------------------
BOUNDS = {
    "Tg":            (-200,  500,  "°C   glass transition"),
    "Td":            (100,   900,  "°C   decomposition onset"),
    "Tm":            (-150,  600,  "°C   melting point"),
    "LOI":           (0,     100,  "%    limiting oxygen index is a percentage"),
    "EPS":           (1.0,   100,  "     relative permittivity cannot be below 1 (vacuum)"),
    "Refractive":    (1.20,  2.00, "     no polymer is below ~1.29 or above ~1.8"),
    "GasPerma":      (0,     1e7,  "Barrer  permeability cannot be negative"),
    "Solubility":    (0,     30,   "(cal/cm3)^0.5"),
    "Hansen":        (0,     60,   "MPa^0.5"),
    "ThermalCond":   (0,     5,    "W/m/K  polymers are ~0.1-0.5"),
    "BandgapBulk":   (0,     15,   "eV   a band gap cannot be negative"),
    "BandgapChain":  (0,     15,   "eV"),
    "BandgapCrystal": (0,    15,   "eV"),
    "CTE":           (-50,   1000, "ppm/K"),
    "Degradability": (-1,    5,    ""),
}


def bound_violations(smi, preds):
    out = []
    for p, v in preds.items():
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            out.append((p, v, "non-finite value")); continue
        b = BOUNDS.get(p)
        if not b:
            continue
        lo, hi, desc = b
        if not (lo <= v <= hi):
            out.append((p, round(float(v), 3), f"outside [{lo}, {hi}] — {desc}"))
    return out


def consistency_violations(smi, preds):
    """Relations that must hold for a real polymer."""
    out = []
    tg, tm, td = preds.get("Tg"), preds.get("Tm"), preds.get("Td")
    amorph = cr.likely_amorphous(smi)
    if tg is not None and tm is not None and amorph is False and tg >= tm:
        out.append(("Tg/Tm", f"Tg {tg:.0f} >= Tm {tm:.0f}", "semi-crystalline needs Tg < Tm"))
    if tg is not None and td is not None and tg > td:
        out.append(("Tg/Td", f"Tg {tg:.0f} > Td {td:.0f}",
                    "a polymer cannot soften above the temperature at which it decomposes"))
    if tm is not None and td is not None and amorph is False and tm > td:
        out.append(("Tm/Td", f"Tm {tm:.0f} > Td {td:.0f}", "cannot melt above decomposition"))
    ri, eps = preds.get("Refractive"), preds.get("EPS")
    if ri is not None and eps is not None and eps + 0.35 < ri * ri:
        # static permittivity must be >= the optical value n^2 (small tolerance for model noise)
        out.append(("EPS/RI", f"EPS {eps:.2f} < n^2 {ri*ri:.2f}",
                    "static permittivity cannot sit below the optical value n^2"))
    return out


def main(n_ga=250):
    print("=" * 96)
    print("CHEMICAL / PHYSICAL VALIDITY AUDIT")
    print("=" * 96)
    props = list(A.models.keys())

    # ---------- canonical polymers ----------
    KNOWN = {"PE": "*CCCC*", "PP": "*CC(C)*", "PS": "*CC(c1ccccc1)*",
             "PMMA": "*CC(C)(C(=O)OC)*", "PVC": "*CC(Cl)*", "PTFE": "*C(F)(F)C(F)(F)*",
             "PET": "*OCCOC(=O)c1ccc(C(=O)*)cc1", "PA6": "*NCCCCCC(=O)*",
             "PC": "*OC(=O)Oc1ccc(C(C)(C)c2ccc(*)cc2)cc1", "PDMS": "*[Si](C)(C)O*",
             "PEO": "*CCO*", "PLA": "*OC(C)C(=O)*", "POM": "*COC*",
             "PSU": "*Oc1ccc(cc1)S(=O)(=O)c1ccc(*)cc1", "PAN": "*CC(C#N)*"}
    print(f"\n[1] CANONICAL POLYMERS ({len(KNOWN)})")
    nb = nc = 0
    for n, s in KNOWN.items():
        pr = A.compute_preds(s, A.models, props) or {}
        bv, cv = bound_violations(s, pr), consistency_violations(s, pr)
        nb += len(bv); nc += len(cv)
        for p, v, why in bv:
            print(f"    BOUND  {n:5} {p:14} = {v}  {why}")
        for topic, what, why in cv:
            print(f"    CONSIS {n:5} {topic:14} {what}  ({why})")
    print(f"    -> {nb} bound violations, {nc} consistency violations")

    # ---------- GA-generated structures ----------
    print(f"\n[2] GA-GENERATED STRUCTURES (target {n_ga})")
    RANGES = {"Tg": {"min": -150., "max": 400.}, "LOI": {"min": 15., "max": 100.},
              "Td": {"min": 150., "max": 700.}, "GasPerma": {"min": 0., "max": 5000.},
              "Refractive": {"min": 1.2, "max": 1.8}, "EPS": {"min": 1.5, "max": 12.}}
    random.seed(3)
    base = [A.smiles_to_selfies_safe(x) for x in _SEEDS]; base = [b for b in base if b]
    pool = set()
    for _ in range(6):
        tp = random.sample(list(RANGES), 3)
        tg = {p: random.uniform(RANGES[p]["min"], RANGES[p]["max"]) for p in tp}
        seed = sga.build_seed_population(tp, tg, RANGES, base)
        bias = sga.goal_bias_from_targets(tp, tg, RANGES)
        inds = [[random.choice(seed)] for _ in range(40)]
        for _g in range(6):
            for ind in inds:
                A.generate_offspring(ind, seed, prop_bias=bias)
                sm = A.selfies_to_smiles_safe(ind[0])
                if sm and sga.is_valid_polymer_smiles(sm):
                    pool.add(sm)
        if len(pool) >= n_ga:
            break
    pool = sorted(pool)[:n_ga]
    print(f"    generated {len(pool)} valid structures")

    nb = nc = 0
    bad_bounds, bad_cons = {}, {}
    charged = unsane = 0
    for sm in pool:
        pr = A.compute_preds(sm, A.models, props) or {}
        for p, v, why in bound_violations(sm, pr):
            nb += 1; bad_bounds.setdefault(p, []).append((sm, v, why))
        for topic, what, why in consistency_violations(sm, pr):
            nc += 1; bad_cons.setdefault(topic, []).append((sm, what))
        m = Chem.MolFromSmiles(sm.replace("*", "[H]"))
        if m is not None and Chem.GetFormalCharge(m) != 0:
            charged += 1
        if not sga.is_chemically_sane(sm):
            unsane += 1
    print(f"    charged repeat units      : {charged}")
    print(f"    failing the sanity filter : {unsane}")
    print(f"    BOUND violations          : {nb}")
    for p, lst in sorted(bad_bounds.items(), key=lambda kv: -len(kv[1])):
        print(f"        {p:14} {len(lst):4}  e.g. {lst[0][1]}  ({lst[0][2]})")
        for sm, v, _w in lst[:2]:
            print(f"            {sm[:66]}")
    print(f"    CONSISTENCY violations    : {nc}")
    for t, lst in sorted(bad_cons.items(), key=lambda kv: -len(kv[1])):
        print(f"        {t:14} {len(lst):4} ({100*len(lst)/max(1,len(pool)):.1f}%)  e.g. {lst[0][1]}")
        for sm, w in lst[:2]:
            print(f"            {sm[:66]}")

    # ---------- retrosynthesis monomers ----------
    print(f"\n[3] RETROSYNTHESIS MONOMERS")
    nroute = nmon = badmon = 0
    for sm in pool:
        r = retro.retro_decompose(sm)
        if not r:
            continue
        nroute += 1
        for mono in r[0]["monomers"]:
            nmon += 1
            mm = Chem.MolFromSmiles(mono)
            if mm is None:
                badmon += 1; print(f"    UNPARSEABLE monomer {mono}"); continue
            try:
                Chem.SanitizeMol(mm)
            except Exception as e:
                badmon += 1; print(f"    UNSANITISABLE {mono}: {str(e)[:50]}"); continue
            if Chem.GetFormalCharge(mm) != 0:
                badmon += 1; print(f"    CHARGED monomer {mono}")
    print(f"    {nroute} routes, {nmon} monomers, {badmon} invalid  "
          f"({100*(nmon-badmon)/max(1,nmon):.1f}% clean)")

    # ---------- chemist review coverage ----------
    print(f"\n[4] CHEMIST REVIEW on the GA pool")
    lv = {"error": 0, "warn": 0, "info": 0}
    flagged = 0
    for sm in pool:
        pr = A.compute_preds(sm, A.models, props) or {}
        notes = cr.review(sm, pr)
        if any(n["level"] == "error" for n in notes):
            flagged += 1
        for n in notes:
            lv[n["level"]] += 1
    print(f"    structures with a hard ERROR : {flagged} ({100*flagged/max(1,len(pool)):.1f}%)")
    print(f"    notes: {lv['error']} error / {lv['warn']} warn / {lv['info']} info")
    print("\n" + "=" * 96)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 250)

"""
golden_set.py — OUT-OF-SAMPLE verification of the Polsen predictors on canonical polymers.

For the ~13 properties that xtb cannot verify per-polymer (bulk / thermal / transport),
this is the honest confidence check: predict textbook polymers with known handbook values
and measure the error. Held-out ML R^2 is optimistic (same data distribution); real, named
polymers test whether a prediction you'd actually trust is trustworthy.

Reference values are approximate handbook figures (Polymer Handbook / van Krevelen / CROW
polymerdatabase.com / Bicerano). Tg, Tm in deg C; RI dimensionless; Solubility = Hildebrand
delta in (cal/cm3)^0.5; LOI in %; EPS = static dielectric constant. None = not applicable/unknown.
Treat MAE as "typical error to expect for a NEW polymer's prediction of this property".

    python validation/golden_set.py
"""
import sys, types, warnings, os
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)

# ------------------------------------------------------------------ load appv3 headlessly
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
st.number_input = lambda *a, **k: k.get("value", 0.0)
st.stop = lambda *a, **k: None; st.set_page_config = lambda *a, **k: None; st.sidebar = st; st.__path__ = []
_cc = _St("streamlit.components"); _cc.__path__ = []; _v = _St("streamlit.components.v1"); _cc.v1 = _v; st.components = _cc
sys.modules["streamlit"] = st; sys.modules["streamlit.components"] = _cc; sys.modules["streamlit.components.v1"] = _v
import datasets as d
d.load_dataset = lambda *a, **k: type("D", (), {"to_pandas": lambda s: pd.DataFrame({"p_smiles": ["*CC(c1ccccc1)*"]})})()
sys.path.insert(0, PROJ)
import importlib.util
spec = importlib.util.spec_from_file_location("appv3", os.path.join(PROJ, "app.py"))
A = importlib.util.module_from_spec(spec); sys.modules["appv3"] = A; spec.loader.exec_module(A)

# ------------------------------------------------------------------ golden reference table
#   name : (SMILES,                                          Tg,   Tm,   RI,   Solub, LOI,  EPS)
GOLD = {
    "PE":        ("*CCCC*",                                  -120,  135, 1.51,  8.0,  17,  2.3),
    "PP":        ("*CC(C)CC(C)*",                             -10,  165, 1.49,  8.0,  18,  2.2),
    "PS":        ("*CC(c1ccccc1)*",                           100, None, 1.59,  9.1,  18,  2.5),
    "PMMA":      ("*CC(C)(C(=O)OC)*",                         105, None, 1.49,  9.3,  17,  3.0),
    "PVC":       ("*CC(Cl)CC(Cl)*",                            80, None, 1.54,  9.6,  45,  3.4),
    "PTFE":      ("*C(F)(F)C(F)(F)*",                        None,  327, 1.35,  6.2,  95,  2.1),
    "PVDF":      ("*CC(F)(F)*",                               -40,  170, 1.42, None,  44,  9.0),
    "PAN":       ("*CC(C#N)*",                                 95,  317, 1.52, 12.5, None, 4.0),
    "PC":        ("*OC(=O)Oc1ccc(C(C)(C)c2ccc(*)cc2)cc1",     147, None, 1.585, 9.8,  25,  3.0),
    "PET":       ("*OCCOC(=O)c1ccc(C(=O)*)cc1",               75,  265, 1.57, 10.7,  21,  3.0),
    "Nylon6":    ("*NCCCCCC(=O)*",                             50,  220, 1.53, 13.6,  21,  3.8),
    "PDMS":      ("*[Si](C)(C)O*",                           -125, None, 1.43,  7.4, None, 2.7),
    "PEO":       ("*CCOCCO*",                                 -60,   65, 1.46,  9.9, None, 5.0),
    "PVA":       ("*CC(O)CC(O)*",                              85,  230, 1.50, 12.6, None, 2.0),
    "PLA":       ("*OC(C)C(=O)*",                              60,  170, 1.45,  9.6, None, 3.0),
    "PCL":       ("*OCCCCCC(=O)*",                            -60,   60, 1.47,  9.2, None, 3.0),
    "Polysulfone": ("*Oc1ccc(cc1)S(=O)(=O)c1ccc(*)cc1",       185, None, 1.63, 10.6,  30,  3.5),
    "POM":       ("*COC*",                                    -60,  175, 1.48, 11.0, None, 3.1),
}
COLS = ["Tg", "Tm", "Refractive", "Solubility", "LOI", "EPS"]
IDX = {c: i + 1 for i, c in enumerate(COLS)}
# "OFF" threshold per property (a prediction beyond this is a real miss, not scatter)
OFF = {"Tg": 40, "Tm": 45, "Refractive": 0.08, "Solubility": 2.0, "LOI": 12, "EPS": 1.5}

print(f"[golden-set] {len(GOLD)} canonical polymers x {len(COLS)} properties\n")
from scipy.stats import spearmanr
rows = []
for name, tup in GOLD.items():
    smi = tup[0]
    if Chem.MolFromSmiles(smi.replace("*", "[H]")) is None:
        print(f"  !! {name}: unparseable SMILES {smi}"); continue
    preds = A.compute_preds(smi, A.models, COLS) or {}
    rows.append({"polymer": name, **{c: (tup[IDX[c]], preds.get(c)) for c in COLS}})

print(f"{'property':<12} {'n':>2}  {'MAE':>7}  {'Spearman':>8}   worst misses")
print("-" * 78)
summary = []
for c in COLS:
    pairs = [(r["polymer"], r[c][0], r[c][1]) for r in rows
             if r[c][0] is not None and r[c][1] is not None]
    if len(pairs) < 3:
        continue
    lit = np.array([p[1] for p in pairs], float); pred = np.array([p[2] for p in pairs], float)
    mae = np.mean(np.abs(pred - lit)); rho = spearmanr(lit, pred).correlation
    order = sorted(pairs, key=lambda p: -abs(p[2] - p[1]))
    worst = ", ".join(f"{n}({pr:.1f}v{lt:.1f})" for n, lt, pr in order[:2])
    print(f"{c:<12} {len(pairs):>2}  {mae:>7.2f}  {rho:>+8.2f}   {worst}")
    summary.append((c, len(pairs), mae, rho))

print("\nPer-property detail:")
for c in COLS:
    pairs = [(r["polymer"], r[c][0], r[c][1]) for r in rows
             if r[c][0] is not None and r[c][1] is not None]
    if len(pairs) < 3: continue
    print(f"\n  === {c}  (unit: {A.prop_unit(c) or '-'}) ===")
    for n, lt, pr in pairs:
        d = pr - lt; flag = "  <-- OFF" if abs(d) > OFF[c] else ""
        print(f"    {n:<12} lit={lt:>7.2f}  pred={pr:>7.2f}  d={d:>+7.2f}{flag}")

print("\nINTERPRETATION: MAE = typical error to expect on a new polymer for that property.")
print("High Spearman + low MAE = trust the absolute value. High Spearman + high MAE = trust")
print("the ranking/direction but not the absolute number. Low Spearman = ranking-only at best.")

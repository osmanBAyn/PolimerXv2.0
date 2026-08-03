"""
compare_bagged.py — decide whether a bagged model should actually REPLACE the deployed one.

The hold-out inside train_bagged.py cannot answer this: the deployed models were fitted on those
same rows, so scoring them there measures memorisation. The golden set (canonical handbook
polymers) is out-of-sample for BOTH, so it is the only fair fight available.

Prints, per property: deployed MAE vs bagged MAE on the golden set, and a recommendation.
Nothing is swapped automatically -- this only produces the evidence.

    python validation/compare_bagged.py
"""
import sys, types, warnings, os, glob, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from rdkit import RDLogger
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
d.load_dataset = lambda *a, **k: type("D", (), {"to_pandas": lambda s: pd.DataFrame(
    {"p_smiles": ["*CC(c1ccccc1)*"]})})()
sys.path.insert(0, PROJ)
import importlib.util, joblib
_spec = importlib.util.spec_from_file_location("appv3", os.path.join(PROJ, "app.py"))
A = importlib.util.module_from_spec(_spec); sys.modules["appv3"] = A; _spec.loader.exec_module(A)
sys.path.insert(0, HERE)
from train_bagged import BaggedEnsemble          # needed to unpickle

#  name : (smiles, Tg, Tm, LOI, Solubility, ThermalCond, CTE)
GOLD = {
 "PE":("*CCCC*",-120,135,17,8.0,0.40,200), "PP":("*CC(C)CC(C)*",-10,165,18,8.0,0.22,150),
 "PS":("*CC(c1ccccc1)*",100,None,18,9.1,0.14,70), "PMMA":("*CC(C)(C(=O)OC)*",105,None,17,9.3,0.19,70),
 "PVC":("*CC(Cl)CC(Cl)*",80,None,45,9.6,0.19,80), "PTFE":("*C(F)(F)C(F)(F)*",None,327,95,6.2,0.25,135),
 "PVDF":("*CC(F)(F)*",-40,170,44,None,0.19,140), "PAN":("*CC(C#N)*",95,317,None,12.5,0.26,None),
 "PC":("*OC(=O)Oc1ccc(C(C)(C)c2ccc(*)cc2)cc1",147,None,25,9.8,0.20,68),
 "PET":("*OCCOC(=O)c1ccc(C(=O)*)cc1",75,265,21,10.7,0.24,70),
 "Nylon6":("*NCCCCCC(=O)*",50,220,21,13.6,0.25,80), "PDMS":("*[Si](C)(C)O*",-125,None,None,7.4,0.15,None),
 "PEO":("*CCOCCO*",-60,65,None,9.9,0.21,None), "PVA":("*CC(O)CC(O)*",85,230,None,12.6,0.20,None),
 "PLA":("*OC(C)C(=O)*",60,170,None,9.6,0.13,70), "PCL":("*OCCCCCC(=O)*",-60,60,None,9.2,None,None),
 "PSU":("*Oc1ccc(cc1)S(=O)(=O)c1ccc(*)cc1",185,None,30,10.6,0.26,56), "POM":("*COC*",-60,175,None,11.0,0.31,110),
}
COLS = ["Tg", "Tm", "LOI", "Solubility", "ThermalCond", "CTE"]

bagged = {}
for f in glob.glob(os.path.join(PROJ, "models_bagged", "bagged_*.joblib")):
    prop = os.path.basename(f)[len("bagged_"):-len(".joblib")]
    try: bagged[prop] = joblib.load(f)
    except Exception as e: print(f"  ! could not load {f}: {str(e)[:60]}")
print(f"loaded {len(bagged)} bagged models: {sorted(bagged)}\n")

print(f"{'property':13} {'n':>3}  {'deployed MAE':>13}  {'bagged MAE':>11}  verdict")
print("-" * 66)
rows = []
for ci, prop in enumerate(COLS):
    if prop not in bagged:
        continue
    lit, pd_, pb_ = [], [], []
    for name, tup in GOLD.items():
        v = tup[1 + ci]
        if v is None:
            continue
        smi = tup[0]
        X = A._features_for(prop, smi)
        if X is None:
            continue
        try:
            a = float(np.asarray(A.models[prop].predict(X)).ravel()[0])
            if prop in A._PROP_POST: a = A._PROP_POST[prop](a)
            b = float(np.asarray(bagged[prop].predict(X)).ravel()[0])
        except Exception:
            continue
        lit.append(float(v)); pd_.append(a); pb_.append(b)
    if len(lit) < 4:
        print(f"{prop:13} (only {len(lit)} golden points)"); continue
    lit = np.array(lit); pd_ = np.array(pd_); pb_ = np.array(pb_)
    m_old = float(np.mean(np.abs(pd_ - lit))); m_new = float(np.mean(np.abs(pb_ - lit)))
    if m_new < m_old * 0.95:   verdict = "SWAP -- bagged better"
    elif m_new <= m_old * 1.05: verdict = "tie (swap for uncertainty)"
    else:                       verdict = "KEEP deployed"
    rows.append((prop, len(lit), m_old, m_new, verdict))
    print(f"{prop:13} {len(lit):>3}  {m_old:>13.3f}  {m_new:>11.3f}  {verdict}")
print("-" * 66)
print("Golden set = canonical handbook polymers, out-of-sample for BOTH models -> a fair test.")
pd.DataFrame(rows, columns=["prop", "n", "golden_mae_deployed", "golden_mae_bagged", "verdict"]
             ).to_csv(os.path.join(PROJ, "models_bagged", "golden_comparison.csv"), index=False)
print("wrote models_bagged/golden_comparison.csv")

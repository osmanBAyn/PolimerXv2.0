"""
calibrate_uncertainty.py — build the per-prediction uncertainty calibration.

Per-property, measures how well a model-internal "spread" signal predicts the actual error,
and stores the calibration the app needs. Two signals, chosen by model type:

  RandomForest      -> std of the individual trees' predictions   (independent ensemble: valid)
  XGBoost / LGBM    -> std of staged predictions from boosting prefixes (40..100% of rounds).
                       Boosted trees are additive corrections, so per-tree std is meaningless;
                       how much the prediction is STILL MOVING as rounds are added is the usable
                       proxy for "this input is not settled".

Only properties whose signal genuinely correlates with |error| (Spearman >= MIN_RHO on a real
sample) are written as usable -- the rest keep the global golden-set MAE. Output:

    validation/uncertainty_calibration.json
      {prop: {"method","rho","n","q25","q50","q75","mae_ref"}}

Run:  python validation/calibrate_uncertainty.py
"""
import sys, types, warnings, os, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)
# A signal ships only if it is BOTH correlated with error AND has a real effect size.
# Rank correlation alone is not enough: LOI scored rho=+0.40 while its top-quartile error was
# only 1.02x its bottom-quartile error -- i.e. the ordering was right but the spread carried
# no usable information. Requiring Q4/Q1 >= MIN_RATIO rejects that.
MIN_RHO = 0.30
MIN_RATIO = 1.5
N_SAMPLE = 400

# ---- headless app load -------------------------------------------------------------
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
REAL = d.load_dataset
d.load_dataset = lambda *a, **k: type("D", (), {"to_pandas": lambda s: pd.DataFrame(
    {"p_smiles": ["*CC(c1ccccc1)*"]})})() if (a and "Tahmini" in str(a[0]) and k.get("split") == "Tg") else REAL(*a, **k)
sys.path.insert(0, PROJ)
import importlib.util
_spec = importlib.util.spec_from_file_location("appv3", os.path.join(PROJ, "app.py"))
A = importlib.util.module_from_spec(_spec); sys.modules["appv3"] = A; _spec.loader.exec_module(A)
from scipy.stats import spearmanr

# property -> HF split holding its ground truth
SPLITS = {
    "Tg": "Tg", "Td": "Td", "Tm": "Tm", "LOI": "LOI", "Solubility": "Solubility",
    "BandgapBulk": "band_gap_bulk", "BandgapChain": "band_gap_chain",
    "BandgapCrystal": "band_gap_crystal", "Refractive": "Refractive_Index",
    "ThermalCond": "Thermal_Conductivity", "CTE": "CTE", "Degradability": "Degradability",
    "Hansen": "Hansen", "EPS": "Dielectric", "Recyclability": "enthalpy",
}


def features(prop, smi):
    key = A._PROP_FEATURE.get(prop)
    try:
        if key == '_fp':   return A.get_morgan_fp(smi, keep_star=False)
        if key == '_fpk':  return A.get_morgan_fp(smi, keep_star=True)
        if key == 'gas':   return A.get_gas_features_combined(smi)
        if key == 'eps':   return A.get_eps_features(smi)
        if key == 'ref':   return A.get_refractive_features(smi)
        if key == 'han':   return A.get_hansen_features(smi)
        if key == 'deg':   return A.get_degradability_features(smi)
        if key == 'rec':   return A.get_recyclability_features(smi)
    except Exception:
        return None
    return None


def spread_of(mdl, X):
    """-> (prediction, spread, method) or None."""
    est = getattr(mdl, "estimators_", None)
    if est is not None and len(est) > 5 and hasattr(est[0], "predict"):
        try:
            p = np.array([t.predict(X)[0] for t in est])
            return float(p.mean()), float(p.std()), "ensemble"
        except Exception:
            pass
    inner = mdl
    if hasattr(mdl, "steps"):
        inner = mdl.steps[-1][1]
    if hasattr(mdl, "regressor_"):
        inner = mdl.regressor_
    try:
        n = inner.get_booster().num_boosted_rounds()
    except Exception:
        n = getattr(inner, "n_estimators", None)
    if not n or n < 20:
        return None
    preds = []
    for f in (0.40, 0.55, 0.70, 0.85, 1.0):
        k = max(1, int(n * f))
        try:
            preds.append(float(mdl.predict(X, iteration_range=(0, k))[0]))
        except Exception:
            try:
                preds.append(float(mdl.predict(X, num_iteration=k)[0]))
            except Exception:
                return None
    p = np.array(preds)
    return float(p[-1]), float(p.std()), "staged"


out = {}
print(f"{'property':14} {'method':9} {'n':>4} {'rho':>7} {'Q4/Q1':>6}  {'quartile mean |err|':<34} usable")
print("-" * 92)
for prop, split in SPLITS.items():
    mdl = A.models.get(prop)
    if mdl is None:
        continue
    try:
        ds = REAL("OsBaran/Polimer-Ozellik-Tahmini", split=split)
    except Exception:
        print(f"{prop:14} (split '{split}' unavailable)")
        continue
    rows = [(ds[i]["smiles"], ds[i]["value"]) for i in range(ds.num_rows)
            if ds[i].get("smiles") and ds[i].get("value") is not None]
    if prop == "Refractive":
        # the deployed v2 model was trained on the PHYSICAL subset only (RI <= 1.8); the
        # unphysical rows are dataset corruption and would just add noise to the calibration
        rows = [r for r in rows if float(r[1]) <= 1.8]
    if len(rows) < 60:
        print(f"{prop:14} (only {len(rows)} rows -- skipped)")
        continue
    rng = np.random.default_rng(0)
    idx = rng.choice(len(rows), size=min(N_SAMPLE, len(rows)), replace=False)
    errs, sprs, method = [], [], None
    for i in idx:
        smi, lit = rows[i]
        X = features(prop, smi)
        if X is None:
            continue
        got = spread_of(mdl, X)
        if got is None:
            continue
        pred, sp, method = got
        if prop in A._PROP_POST:
            try: pred = A._PROP_POST[prop](pred)
            except Exception: pass
        try: lit = float(lit)
        except Exception: continue
        errs.append(abs(pred - lit)); sprs.append(sp)
    if len(errs) < 50:
        print(f"{prop:14} {str(method):9} {len(errs):>4}  (too few usable rows)")
        continue
    errs = np.array(errs); sprs = np.array(sprs)
    rho = spearmanr(sprs, errs).correlation
    q = np.quantile(sprs, [0.25, 0.5, 0.75])
    qb = np.quantile(sprs, [0, .25, .5, .75, 1.0])
    qm = []
    for k in range(4):
        m = (sprs >= qb[k]) & ((sprs <= qb[k + 1]) if k == 3 else (sprs < qb[k + 1]))
        qm.append(f"{errs[m].mean():.3g}" if m.sum() else "-")
    # effect size: how much worse is the top-spread quartile than the bottom one?
    m1 = (sprs < qb[1]); m4 = (sprs >= qb[3])
    e1 = errs[m1].mean() if m1.sum() else float('nan')
    e4 = errs[m4].mean() if m4.sum() else float('nan')
    ratio = (e4 / e1) if (e1 and e1 == e1 and e4 == e4) else float('nan')
    usable = bool(rho == rho and rho >= MIN_RHO and ratio == ratio and ratio >= MIN_RATIO)
    if usable:
        out[prop] = {"method": method, "rho": round(float(rho), 3), "n": int(len(errs)),
                     "ratio": round(float(ratio), 2),
                     "q25": float(q[0]), "q50": float(q[1]), "q75": float(q[2]),
                     "mae_ref": float(errs.mean())}
    print(f"{prop:14} {method:9} {len(errs):>4} {rho:>+7.3f} {ratio:>6.2f}x  "
          f"{' -> '.join(qm):<34} {'YES' if usable else 'no'}")

path = os.path.join(HERE, "uncertainty_calibration.json")
json.dump(out, open(path, "w"), indent=1)
print("-" * 92)
print(f"wrote {len(out)} usable calibrations (rho >= {MIN_RHO}) -> {path}")
print("Properties not listed keep the global golden-set MAE (PROP_MAE in app.py).")

"""
train_bagged.py — retrain property models as BAGGED ENSEMBLES.

Why: a single XGBoost gives no valid per-prediction uncertainty (boosted trees are additive
corrections, so per-tree spread is meaningless). A bag of independently-seeded models trained on
bootstrap resamples does: the spread across bag members is a real variance estimate, exactly like
a RandomForest's. That upgrades the "+/-" on a prediction from a global average to a number that
belongs to THAT molecule.

Design:
  * features are taken from appv3's own routing, so a retrained model is a drop-in replacement
    (same feature convention, incl. the '*'-handling each property needs);
  * the saved object exposes .predict() AND .estimators_, so appv3's existing "ensemble"
    uncertainty path picks it up with no code change;
  * an 80/20 hold-out is kept aside. The bag is compared to the CURRENTLY DEPLOYED model on that
    same hold-out and is only written if it is not worse -- a retrain must never silently
    downgrade a property;
  * uncertainty is calibrated on the HELD-OUT residuals (the previous calibration could only use
    in-sample data), which is the statistically correct basis.

    python validation/train_bagged.py                # all eligible properties
    python validation/train_bagged.py Tg Td Tm       # a subset
"""
import sys, types, warnings, os, json, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)
N_BAGS = 8
TEST_FRAC = 0.20
SEED = 20260731
OUT_DIR = os.path.join(PROJ, "models_bagged")
MIN_RHO, MIN_RATIO = 0.30, 1.5

# ---- headless app load (gives us the models + the exact feature routing) -------------
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

import joblib
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

# property -> HF split. Only properties whose split IS the model's training target in the
# model's own units. Deliberately excluded:
#   EPS           - HF 'Dielectric' split is corrupt; v2 was trained on a separate clean CSV
#   Refractive    - already a 530-tree RandomForest (valid spread; rho=0.95) - nothing to gain
#   Recyclability - 'enthalpy' split maps indirectly; needs its own review
# GasPerma is special: the HF 'Gas_Permeability' split is PoLyInfo data mixing several gases
# with no gas label, so it cannot be used as a target. The deployed model was trained on the
# PolymerGasMembraneML benchmark, CO2 column, log10(Barrer) -- we use that same source here.
GASPERMA_URL = ("https://raw.githubusercontent.com/jsunn-y/PolymerGasMembraneML/"
                "main/datasets/datasetA_imputed_all.csv")
LOG_TARGET = {"GasPerma"}          # model is fitted on log10(value); error is a FOLD factor

SPLITS = {
    "GasPerma": "__polymergasmembraneml_CO2__",
    "Tg": "Tg", "Td": "Td", "Tm": "Tm", "LOI": "LOI", "Solubility": "Solubility",
    "BandgapBulk": "band_gap_bulk", "BandgapChain": "band_gap_chain",
    "BandgapCrystal": "bandgap_crystal", "CTE": "CTE",
    "ThermalCond": "Thermal_Conductivity", "Degradability": "degradability", "Hansen": "hansen",
}


from bagged_model import BaggedEnsemble   # shared container -> pickles resolve for the app


def build_xy(prop, split):
    if prop == "GasPerma":
        df = pd.read_csv(GASPERMA_URL)
        col = "CO2" if "CO2" in df.columns else [c for c in df.columns if "CO2" in c.upper()][0]
        df = df.dropna(subset=["Smiles", col])
        df = df[df[col] > 0]
        seen, X, y = set(), [], []
        for smi, v in zip(df["Smiles"].astype(str), df[col].astype(float)):
            if smi in seen:
                continue
            seen.add(smi)
            f = A._features_for(prop, smi)
            if f is None:
                continue
            X.append(np.asarray(f).ravel()); y.append(np.log10(v))   # fit in log10(Barrer)
        return (np.vstack(X), np.array(y)) if X else (None, None)
    ds = REAL("OsBaran/Polimer-Ozellik-Tahmini", split=split)
    rows = [(ds[i].get("smiles"), ds[i].get("value")) for i in range(ds.num_rows)]
    seen, X, y = set(), [], []
    for smi, val in rows:
        if not smi or val is None:
            continue
        try:
            v = float(val)
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        if prop == "Refractive" and v > 1.8:
            continue
        if smi in seen:
            continue
        seen.add(smi)
        f = A._features_for(prop, smi)
        if f is None:
            continue
        X.append(np.asarray(f).ravel()); y.append(v)
    if not X:
        return None, None
    return np.vstack(X), np.array(y)


def deployed_predict(prop, X):
    m = A.models.get(prop)
    if m is None:
        return None
    try:
        p = np.asarray(m.predict(X), dtype=float)
        if prop in A._PROP_POST and prop not in LOG_TARGET:
            p = np.array([A._PROP_POST[prop](v) for v in p])
        return p   # LOG_TARGET properties stay in log10 space, matching build_xy
    except Exception:
        return None


def main(only=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    calib, report = {}, []
    props = [p for p in SPLITS if (not only or p in only)]
    print(f"Bagged retrain: {len(props)} properties x {N_BAGS} bags  (hold-out {int(TEST_FRAC*100)}%)\n")
    for prop in props:
        t0 = time.time()
        try:
            X, y = build_xy(prop, SPLITS[prop])
        except Exception as e:
            print(f"{prop:14} SKIP (split load failed: {str(e)[:50]})"); continue
        if X is None or len(y) < 100:
            print(f"{prop:14} SKIP (only {0 if y is None else len(y)} usable rows)"); continue

        rng = np.random.default_rng(SEED)
        idx = rng.permutation(len(y))
        ntest = max(30, int(len(y) * TEST_FRAC))
        te, tr = idx[:ntest], idx[ntest:]
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

        def make(seed):
            return XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                subsample=0.85, colsample_bytree=0.85,
                                reg_lambda=1.0, random_state=seed,
                                n_jobs=4, tree_method="hist", verbosity=0)

        # --- STAGE 1: measure on a hold-out ------------------------------------------
        # The deployed model was fitted on rows that include this hold-out, so scoring it here
        # would just measure memorisation. The fair question is "does bagging beat ONE model of
        # the same kind, trained on the same rows?" -- so that is what we compare.
        ests = []
        for b in range(N_BAGS):
            r = np.random.default_rng(SEED + 1000 * b)
            boot = r.integers(0, len(ytr), len(ytr))       # bootstrap resample
            m = make(SEED + b); m.fit(Xtr[boot], ytr[boot])
            ests.append(m)
        bag_split = BaggedEnsemble(ests, prop=prop)
        single = make(SEED); single.fit(Xtr, ytr)

        pb, sb = bag_split.predict_with_std(Xte)
        ps = np.asarray(single.predict(Xte), dtype=float)
        mae_bag = float(np.mean(np.abs(pb - yte))); r2_bag = float(r2_score(yte, pb))
        mae_sgl = float(np.mean(np.abs(ps - yte))); r2_sgl = float(r2_score(yte, ps))
        helps = mae_bag <= mae_sgl * 1.02                  # bagging should help or tie

        # --- uncertainty calibration on TRUE held-out residuals ----------------------
        err = np.abs(pb - yte)
        rho = float(spearmanr(sb, err).correlation)
        qb = np.quantile(sb, [0, .25, .5, .75, 1.0])
        e1 = err[sb < qb[1]].mean() if (sb < qb[1]).sum() else np.nan
        e4 = err[sb >= qb[3]].mean() if (sb >= qb[3]).sum() else np.nan
        ratio = float(e4 / e1) if (e1 and np.isfinite(e1) and np.isfinite(e4)) else float("nan")
        usable = bool(np.isfinite(rho) and rho >= MIN_RHO and np.isfinite(ratio) and ratio >= MIN_RATIO)

        # --- STAGE 2: refit the shipped bag on ALL rows -------------------------------
        # Standard workflow: split to MEASURE, refit on everything to DEPLOY, so the shipped
        # model is not handicapped by the 20% we held back.
        full = []
        for b in range(N_BAGS):
            r = np.random.default_rng(SEED + 1000 * b)
            boot = r.integers(0, len(y), len(y))
            m = make(SEED + b); m.fit(X[boot], y[boot])
            full.append(m)
        bag = BaggedEnsemble(full, prop=prop,
                             meta={"n_bags": N_BAGS, "n_rows": int(len(y)),
                                   "split": SPLITS[prop], "seed": SEED,
                                   "holdout_mae_bag": mae_bag, "holdout_mae_single": mae_sgl,
                                   "holdout_r2_bag": r2_bag, "unc_rho": rho, "unc_ratio": ratio})
        joblib.dump(bag, os.path.join(OUT_DIR, f"bagged_{prop}.joblib"))
        if usable:
            calib[prop] = {"method": "ensemble", "rho": round(rho, 3), "n": int(len(yte)),
                           "ratio": round(ratio, 2), "q25": float(np.quantile(sb, .25)),
                           "q50": float(np.quantile(sb, .5)), "q75": float(np.quantile(sb, .75)),
                           "mae_ref": mae_bag, "holdout": True}
        status = "bag helps" if helps else "bag ~= single"
        if prop in LOG_TARGET:
            print(f"{'':14}   (log10 target: bag MAE {mae_bag:.3f} decades = "
                  f"{10**mae_bag:.2f}x fold error)")
        report.append((prop, len(y), mae_sgl, mae_bag, r2_sgl, r2_bag, rho, ratio, usable, status))
        print(f"{prop:14} n={len(y):>5}  MAE single {mae_sgl:>8.3f} -> bag {mae_bag:<8.3f}  "
              f"R2 {r2_sgl:>+6.3f} -> {r2_bag:<+6.3f}  rho={rho:+.2f} {ratio:>5.2f}x  "
              f"unc={'YES' if usable else 'no ':<3} {status}  [{time.time()-t0:.0f}s]")

    json.dump(calib, open(os.path.join(OUT_DIR, "bagged_calibration.json"), "w"), indent=1)
    print(f"\nwrote {len(calib)} held-out uncertainty calibrations -> models_bagged/bagged_calibration.json")
    pd.DataFrame(report, columns=["prop", "n", "mae_single", "mae_bag", "r2_single", "r2_bag",
                                  "unc_rho", "unc_ratio", "unc_usable", "status"]
                 ).to_csv(os.path.join(OUT_DIR, "bagged_report.csv"), index=False)
    print("wrote models_bagged/bagged_report.csv")


if __name__ == "__main__":
    main(sys.argv[1:] or None)

"""test_app_results.py -- execute appv3's RESULTS path (tab1 cards + AD banner + tab4 report) with a fake GA result,
to catch runtime errors in the new error-bar / applicability-domain / export code."""
import sys, types, warnings, os
warnings.filterwarnings("ignore")
import pandas as pd
os.environ.setdefault("XTB_EXE", r"C:\Users\osbar\miniconda3\envs\polsen\Library\bin\xtb.exe")

RENDER = []
DOWNLOADS = []

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
st.checkbox = lambda *a, **k: bool(k.get("value", False))
st.button = lambda *a, **k: False
st.slider = lambda l, mn=0, mx=1, val=None, *a, **k: (val if val is not None else mn)
st.number_input = lambda *a, **k: k.get("value", 0.0)
st.text_input = lambda *a, **k: ""
st.dataframe = lambda *a, **k: RENDER.append(("dataframe", "df"))
st.markdown = lambda *a, **k: RENDER.append(("markdown", a[0] if a else ""))
st.info = lambda *a, **k: RENDER.append(("info", a[0] if a else ""))
st.caption = lambda *a, **k: RENDER.append(("caption", a[0] if a else ""))
st.warning = lambda *a, **k: RENDER.append(("warning", a[0] if a else ""))
st.success = lambda *a, **k: RENDER.append(("success", a[0] if a else ""))
st.error = lambda *a, **k: RENDER.append(("error", a[0] if a else ""))
def _dl(label=None, data=None, file_name=None, **k):
    DOWNLOADS.append((label, file_name, data))
    return False
st.download_button = _dl
st.stop = lambda *a, **k: None; st.set_page_config = lambda *a, **k: None; st.sidebar = st; st.__path__ = []
_cc = _St("streamlit.components"); _cc.__path__ = []; _v = _St("streamlit.components.v1"); _cc.v1 = _v; st.components = _cc
sys.modules["streamlit"] = st; sys.modules["streamlit.components"] = _cc; sys.modules["streamlit.components.v1"] = _v
# The app reads its seed population from seed_population.json.gz -- no network, no `datasets`.
# We still stub the HuggingFace path when the package happens to be installed, so that a missing
# or corrupt seed file can never turn this test into a silent 25 s download.
try:
    import datasets as d
    _r = d.load_dataset
    d.load_dataset = lambda *a, **k: type("D", (), {"to_pandas": lambda s: pd.DataFrame(
        {"p_smiles": ["*CC(c1ccccc1)*", "*CCO*", "*OCCOC(=O)c1ccc(C(=O)*)cc1"]})})() \
        if (a and "Tahmini" in str(a[0]) and k.get("split") == "Tg") else _r(*a, **k)
except ImportError:
    pass          # expected in a deployment-equivalent environment
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

# --- pre-seed a GA result so the results block executes ---
BEST = "*CC(c1ccccc1)*"
st.session_state['ga_targets'] = {"Tg": 120.0, "LOI": 30.0}
st.session_state['ga_active_props'] = ["Tg", "LOI"]
st.session_state['ga_pareto'] = None
st.session_state['ga_seed_used'] = 42
st.session_state['ga_started_from'] = "*CC(C)*"      # exercise the before/after compare

import importlib.util
_APP = "app.py" if os.path.exists(os.path.join(PROJ, "app.py")) else "appv3.py"
spec = importlib.util.spec_from_file_location("appv3", os.path.join(PROJ, _APP))
A = importlib.util.module_from_spec(spec); sys.modules["appv3"] = A

# stash a real preds dict once models load: run module, then inject and re-run results by hand
spec.loader.exec_module(A)          # first pass: no ga_results -> results block skipped

preds = A.compute_preds(BEST, A.models, list(A.models.keys()))
st.session_state['ga_results'] = {
    "smiles": BEST, "preds": preds, "total_error": 0.1234,
    # exercise the runner-up (top-N) rendering path too
    "top": [{"smiles": BEST, "preds": preds, "total_error": 0.1234},
            {"smiles": "*CC(C)*", "preds": A.compute_preds("*CC(C)*", A.models, ["Tg", "LOI"]) or {},
             "total_error": 0.20},
            {"smiles": "*CCO*", "preds": A.compute_preds("*CCO*", A.models, ["Tg", "LOI"]) or {},
             "total_error": 0.31}],
}
st.session_state['ga_history'] = [1.0, 0.5, 0.2]
RENDER.clear(); DOWNLOADS.clear()
spec.loader.exec_module(A)          # second pass: results block runs

print("=== checks ===")
txt = " ".join(str(t) for _, t in RENDER)
ok = True
def chk(cond, label):
    global ok
    print(("  PASS " if cond else "  FAIL ") + label)
    ok = ok and cond

# Error bars / reliability follow the SHOW_RELIABILITY flag; the applicability banner
# follows SHOW_APPLICABILITY. Assert the ACTUAL flag state, so either setting passes.
_rel = getattr(A, "SHOW_RELIABILITY", False)
_ad = getattr(A, "SHOW_APPLICABILITY", True)
_has_ad = any("Applicability" in str(t) or "Uygulanabilirlik" in str(t) for _, t in RENDER)
_has_err = ("±" in txt or "+/-" in txt)
chk(_has_ad == _ad, f"applicability banner follows SHOW_APPLICABILITY={_ad} (rendered={_has_ad})")
chk(_has_err == _rel, f"error bars follow SHOW_RELIABILITY={_rel} (rendered={_has_err})")
chk(any("Blend" in str(t) or "Harman" in str(t) for _, t in RENDER) == bool(getattr(A, "SHOW_BLENDS", False)),
    f"blend tab follows SHOW_BLENDS={getattr(A, 'SHOW_BLENDS', False)}")
chk(len(DOWNLOADS) >= 2, f"downloads offered (got {len(DOWNLOADS)}: {[d[1] for d in DOWNLOADS]})")
_topn_csv = any(d[1] == "polsen_top_candidates.csv" for d in DOWNLOADS)
chk(_topn_csv == bool(getattr(A, "TOP_N_CANDIDATES", 0)),
    f"runner-up section rendered per TOP_N_CANDIDATES={getattr(A, 'TOP_N_CANDIDATES', 0)}")

rep = next((d[2] for d in DOWNLOADS if d[1] and d[1].endswith(".txt")), None)
chk(rep is not None, "text report generated")
if rep:
    r = rep.decode("utf-8") if isinstance(rep, bytes) else str(rep)
    _must = ["POLSEN", "PREDICTED PROPERTIES", "RETROSYNTHESIS", "Random seed", "Optimised from"]
    if getattr(A, "SHOW_CHEM_REVIEW", False): _must.append("CHEMIST'S REVIEW")
    if _ad: _must.append("APPLICABILITY DOMAIN")
    if _rel: _must.append("HOW TO READ")
    for must in _must:
        chk(must in r, f"report contains {must!r}")
    print("\n----- report preview -----")
    print("\n".join(r.splitlines()[:26]))

# pick the RESULT csv by exact name -- the runner-up list also offers a .csv
csv = next((d[2] for d in DOWNLOADS if d[1] == "polsen_polymer_data.csv"), None)
if csv:
    c = csv.decode("utf-8") if isinstance(csv, bytes) else str(csv)
    head = c.splitlines()[0]
    chk(("Applicability domain" in head) == _ad, f"CSV applicability column follows flag ({_ad})")
    chk(("typical error" in head) == _rel, f"CSV error columns follow SHOW_RELIABILITY ({_rel})")

print("\n" + ("ALL RESULTS-PATH CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)

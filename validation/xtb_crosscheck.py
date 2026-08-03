"""
xtb_crosscheck.py — Independent quantum validation of Polsen's electronic-property models.

Validates the three ML property models that are physically accessible to a fast
semi-empirical quantum method (GFN2-xTB) against xtb itself, on real polymer repeat
units sampled from OsBaran/Polimer-Ozellik-Tahmini:

    band gap   : xtb HOMO-LUMO gap of n=1,2,3 H-capped oligomers, extrapolated to the
                 infinite chain (gap vs 1/n), compared to the BandgapChain model.
    refractive : Lorentz-Lorenz  (n^2-1)/(n^2+2)  ~  alpha/V  (electronic polariz. density).
    dielectric : Clausius-Mossotti (eps-1)/(eps+2) ~ alpha/V  (electronic part of eps).

Why rank correlation is the metric: GFN2-xTB is semi-empirical, so its ABSOLUTE band gaps
carry a large systematic offset vs the DFT the models were trained on, and alpha/V is only
proportional to the LL/CM functions (no absolute density). What is meaningful is whether xtb
and the model ORDER polymers the same way -> Spearman rho. A high rho means the model's
ranking is reproduced by an independent quantum method it never saw.

Requirements: rdkit, numpy, scipy, datasets, and an xtb executable.
Point at xtb via env var XTB_EXE, or install it:  conda install -c conda-forge xtb
Usage:
    python xtb_crosscheck.py bandgap      # or: ri | eps | all
    python xtb_crosscheck.py all -n 24

NOTE: xtb is used here as an OFFLINE validation tool. It is NOT a deployment dependency —
the Streamlit app never calls it. These numbers are evidence about model trustworthiness.
"""
import sys, types, warnings, random, subprocess, tempfile, os, re, argparse
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
XTB = os.environ.get("XTB_EXE", r"C:\Users\osbar\miniconda3\envs\polsen\Library\bin\xtb.exe")

# ---------------------------------------------------------------- load appv3 headlessly
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
def _load_app():
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
    global REALLOAD; REALLOAD = d.load_dataset
    d.load_dataset = lambda *a, **k: type("D", (), {"to_pandas": lambda s: pd.DataFrame({"p_smiles": ["*CC(c1ccccc1)*"]})})() \
        if (a and "Tahmini" in str(a[0]) and k.get("split") == "Tg") else REALLOAD(*a, **k)
    sys.path.insert(0, PROJ)
    import importlib.util
    spec = importlib.util.spec_from_file_location("appv3", os.path.join(PROJ, "app.py"))
    A = importlib.util.module_from_spec(spec); sys.modules["appv3"] = A; spec.loader.exec_module(A)
    return A

# ---------------------------------------------------------------- xtb runners
def _xyz_from_mol(mol):
    mh = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mh, randomSeed=42) != 0:
        if AllChem.EmbedMolecule(mh, randomSeed=1, useRandomCoords=True) != 0:
            return None, None
    try: AllChem.MMFFOptimizeMolecule(mh, maxIters=500)
    except Exception: pass
    conf = mh.GetConformer()
    lines = [str(mh.GetNumAtoms()), ""]
    for at in mh.GetAtoms():
        p = conf.GetAtomPosition(at.GetIdx())
        lines.append(f"{at.GetSymbol()} {p.x:.6f} {p.y:.6f} {p.z:.6f}")
    return "\n".join(lines), mh

def _run_xtb(xyz):
    with tempfile.TemporaryDirectory() as td:
        open(os.path.join(td, "m.xyz"), "w").write(xyz)
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        # latin-1 + replace: xtb's banner has box-drawing/Greek glyphs that crash cp1254
        return subprocess.run([XTB, "m.xyz", "--gfn", "2", "--sp"], cwd=td, capture_output=True,
                              text=True, timeout=180, env=env, encoding="latin-1", errors="replace").stdout

def xtb_gap(mol):
    xyz, _ = _xyz_from_mol(mol)
    if xyz is None: return None
    m = re.search(r"HOMO-LUMO GAP\s+([-\d.]+)\s+eV", _run_xtb(xyz))
    return float(m.group(1)) if m else None

def xtb_alpha_vol(smi):
    """Electronic polarizability alpha (au) and molecular volume V (A^3) of the H-capped unit."""
    mol = Chem.MolFromSmiles(str(smi).replace("*", "[H]"))
    if mol is None: return None, None
    xyz, mh = _xyz_from_mol(mol)
    if xyz is None: return None, None
    try: vol = AllChem.ComputeMolVolume(mh)
    except Exception: vol = None
    # match "Mol. <alpha>(0) /au : value"; skip the C6AA line ("/au.bohr"). Greek glyph is
    # mangled by latin-1, so anchor on the "/au :" tail rather than the letter.
    m = re.search(r"Mol\.[^:\n]*?/au\s*:\s*([-\d.]+)", _run_xtb(xyz))
    return (float(m.group(1)) if m else None), vol

# ---------------------------------------------------------------- band-gap oligomer chain
def build_oligomer(smi, n):
    unit = Chem.MolFromSmiles(smi)
    if unit is None: return None
    stars = [a.GetIdx() for a in unit.GetAtoms() if a.GetAtomicNum() == 0]
    if len(stars) != 2: return None
    nb = []
    for s in stars:
        ns = unit.GetAtomWithIdx(s).GetNeighbors()
        if not ns: return None
        nb.append(ns[0].GetIdx())
    (head_star, tail_star), (head_nb, tail_nb) = stars, (nb[0], nb[1])
    na = unit.GetNumAtoms()
    combo = Chem.RWMol(unit)
    for _ in range(n - 1): combo.InsertMol(unit)
    rm = []
    for i in range(n):
        off = i * na
        if i < n - 1:
            combo.AddBond(off + tail_nb, (i + 1) * na + head_nb, Chem.BondType.SINGLE)
            rm += [off + tail_star, (i + 1) * na + head_star]
    rm += [head_star, (n - 1) * na + tail_star]
    for idx in sorted(set(rm), reverse=True): combo.RemoveAtom(idx)
    m = combo.GetMol()
    try: Chem.SanitizeMol(m); return m
    except Exception: return None

def extrapolate_gap(smi, ns=(1, 2, 3)):
    xs, ys = [], []
    for n in ns:
        oli = build_oligomer(smi, n)
        if oli is None or oli.GetNumHeavyAtoms() > 90: continue
        g = xtb_gap(oli)
        if g is not None: xs.append(1.0 / n); ys.append(g)
    if len(ys) >= 2:
        a, b = np.polyfit(xs, ys, 1)   # gap = a*(1/n)+b ; b = infinite-chain limit
        return b
    return ys[0] if ys else None

# ---------------------------------------------------------------- sampling helpers
def _sample(A, split, n, prop, vmax=None, hmin=4, hmax=22):
    ds = REALLOAD("OsBaran/Polimer-Ozellik-Tahmini", split=split)
    out = []
    for i in range(ds.num_rows):
        smi, val = ds[i]["smiles"], ds[i]["value"]
        if not smi or smi.count("*") != 2 or val is None: continue
        if vmax is not None and float(val) > vmax: continue
        m = Chem.MolFromSmiles(smi.replace("*", "[H]"))
        if m is None or not (hmin <= m.GetNumHeavyAtoms() <= hmax): continue
        out.append((smi, float(val)))
    out.sort(key=lambda t: t[1])
    idx = np.linspace(0, len(out) - 1, min(n, len(out))).astype(int)
    return len(out), [out[i] for i in idx]

def _report(name, pairs):
    from scipy.stats import spearmanr
    npts = len(pairs[0][1]) if pairs else 0        # data points, not # of comparisons
    print(f"\n================ {name}  (n={npts} polymers) ================")
    for label, x, y in pairs:
        print(f"  {label:<40} Spearman {spearmanr(x, y).correlation:+.2f}")

# ---------------------------------------------------------------- validations
def validate_bandgap(A, n):
    LO, HI = 0.0, 100.0
    tot, sample = _sample(A, "band_gap_chain", n, "BandgapChain")
    print(f"[band gap] {tot} candidates; testing {len(sample)} spanning "
          f"{sample[0][1]:.2f}..{sample[-1][1]:.2f} eV")
    data, model, xtbv = [], [], []
    print(f"{'#':>2} {'data':>5} {'model':>6} {'xtb_inf':>7}  smiles")
    for k, (smi, dv) in enumerate(sample):
        p = A.compute_preds(smi, A.models, ["BandgapChain"])
        mv = p.get("BandgapChain") if p else None
        xv = extrapolate_gap(smi)
        if mv is not None and xv is not None:
            data.append(dv); model.append(mv); xtbv.append(xv)
            print(f"{k:>2} {dv:>5.2f} {mv:>6.2f} {xv:>7.2f}  {smi[:42]}")
    _report("BAND GAP  (eV)", [("model vs dataset", model, data),
                               ("xtb   vs dataset", xtbv, data),
                               ("model vs xtb  (independent)", model, xtbv)])

def validate_polar(A, n, prop_key, split, vmax, func, fname, title):
    tot, sample = _sample(A, split, n, prop_key, vmax=vmax)
    print(f"[{title}] {tot} candidates; testing {len(sample)}")
    model, av, data = [], [], []
    print(f"{'#':>2} {'model':>7} {'a/V':>6}  smiles")
    for k, (smi, dv) in enumerate(sample):
        p = A.compute_preds(smi, A.models, [prop_key]); mv = p.get(prop_key) if p else None
        alpha, vol = xtb_alpha_vol(smi)
        if mv is not None and alpha and vol:
            model.append(mv); av.append(alpha / vol); data.append(dv)
            print(f"{k:>2} {mv:>7.3f} {alpha/vol:>6.3f}  {smi[:42]}")
    _report(f"{title}   (via {fname})",
            [(f"model {fname} vs xtb alpha/V  (independent)", av, [func(x) for x in model]),
             ("model raw vs xtb alpha/V", av, model)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prop", nargs="?", default="all", choices=["bandgap", "ri", "eps", "all"])
    ap.add_argument("-n", type=int, default=22, help="polymers to test per property")
    args = ap.parse_args()
    print(f"[setup] xtb = {XTB} | exists: {os.path.exists(XTB)}")
    if not os.path.exists(XTB):
        print("  !! xtb not found. Set XTB_EXE or `conda install -c conda-forge xtb`."); return
    A = _load_app()
    LL = lambda n: (n * n - 1) / (n * n + 2)
    CM = lambda e: (e - 1) / (e + 2)
    if args.prop in ("bandgap", "all"): validate_bandgap(A, args.n)
    if args.prop in ("ri", "all"):
        validate_polar(A, args.n, "Refractive", "Refractive_Index", 1.8, LL, "Lorentz-Lorenz", "REFRACTIVE INDEX")
    if args.prop in ("eps", "all"):
        validate_polar(A, args.n, "EPS", "Refractive_Index", None, CM, "Clausius-Mossotti", "DIELECTRIC (EPS)")

if __name__ == "__main__":
    main()

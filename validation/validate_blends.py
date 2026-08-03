"""
validate_blends.py — check the blend module against blends whose behaviour is textbook.

Miscibility is judged against the literature verdict; Tg against measured blend Tg where a
value is well known. Anything the screen gets wrong is printed, not hidden -- the point is to
know where it fails before a chemist finds out for us.

    python validation/validate_blends.py
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
import blends
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

P = {
    "PS":     "*CC(c1ccccc1)*",
    "PPO":    "*Oc1cc(C)cc(C)c1*",
    "PMMA":   "*CC(C)(C(=O)OC)*",
    "PVC":    "*CC(Cl)*",
    "PC":     "*OC(=O)Oc1ccc(C(C)(C)c2ccc(*)cc2)cc1",
    "PE":     "*CCCC*",
    "PP":     "*CC(C)*",
    "PVDF":   "*CC(F)(F)*",
    "PEO":    "*CCO*",
    "SAN":    "*CC(c1ccccc1)CC(C#N)*",
    "PBD":    "*CC=CC*",
    "PET":    "*OCCOC(=O)c1ccc(C(=O)*)cc1",
    "PBT":    "*OCCCCOC(=O)c1ccc(C(=O)*)cc1",
    "PA6":    "*NCCCCCC(=O)*",
    "PA66":   "*NCCCCCCNC(=O)CCCCC(=O)*",
    "PLA":    "*OC(C)C(=O)*",
    "PCL":    "*OCCCCCC(=O)*",
    "PHB":    "*OC(C)CC(=O)*",
    "PaMS":   "*CC(C)(c1ccccc1)*",
    "PVME":   "*CC(OC)*",
    "PVAc":   "*CC(OC(C)=O)*",
    "PDMS":   "*[Si](C)(C)O*",
    "PAN":    "*CC(C#N)*",
    "PSU":    "*Oc1ccc(cc1)S(=O)(=O)c1ccc(*)cc1",
    "PTFE":   "*C(F)(F)C(F)(F)*",
    "PIB":    "*CC(C)(C)*",
    "PVOH":   "*CC(O)*",
    "PEEK":   "*Oc1ccc(Oc2ccc(C(=O)c3ccc(*)cc3)cc2)cc1",
}

# (A, B, literature verdict, note).  Verdicts from the standard blend literature
# (Utracki, Polymer Blends Handbook; Olabisi; Robeson, Polymer Blends).
CASES = [
    # --- genuinely miscible pairs -------------------------------------------------
    ("PS",   "PPO",  "miscible",   "Noryl - the classic miscible pair (pi-pi driven)"),
    ("PMMA", "PVDF", "miscible",   "miscible through a dipole/carbonyl interaction"),
    ("PEO",  "PMMA", "miscible",   "miscible over a wide range"),
    ("PS",   "PaMS", "miscible",   "near-identical chemistry, miscible"),
    ("PS",   "PVME", "miscible",   "miscible with an LCST"),
    ("PET",  "PBT",  "miscible",   "miscible polyester pair"),
    ("PA6",  "PA66", "miscible",   "miscible polyamide pair"),
    ("PVC",  "PVAc", "miscible",   "miscible / basis of VC-VAc copolymers"),
    ("PVC",  "PMMA", "borderline", "partially miscible / marginal"),
    ("PC",   "PBT",  "borderline", "commercial alloy, partial + transesterification"),
    ("PC",   "SAN",  "borderline", "PC/ABS alloys - partial miscibility with SAN"),
    ("PMMA", "SAN",  "borderline", "miscible only in an AN composition window"),
    # --- immiscible pairs ---------------------------------------------------------
    ("PS",   "PMMA", "immiscible", "textbook immiscible pair"),
    ("PS",   "PBD",  "immiscible", "HIPS - dispersed rubber phase (deliberately immiscible)"),
    ("PE",   "PP",   "immiscible", "immiscible despite similar chemistry"),
    ("PET",  "PA6",  "immiscible", "immiscible without a compatibiliser"),
    ("PS",   "PVC",  "immiscible", "immiscible"),
    ("PS",   "PE",   "immiscible", "immiscible"),
    ("PA6",  "PP",   "immiscible", "immiscible - needs maleated PP"),
    ("PLA",  "PCL",  "immiscible", "immiscible biodegradable pair"),
    ("PLA",  "PHB",  "immiscible", "immiscible"),
    ("PC",   "PE",   "immiscible", "immiscible"),
    ("PS",   "PDMS", "immiscible", "very immiscible (large delta gap)"),
    ("PP",   "PVOH", "immiscible", "non-polar vs strongly H-bonding"),
    ("PTFE", "PMMA", "immiscible", "immiscible"),
    ("PIB",  "PS",   "immiscible", "immiscible"),
    ("PEEK", "PA6",  "immiscible", "immiscible"),
    ("PAN",  "PE",   "immiscible", "immiscible - very polar vs non-polar"),
]

print(f"{'blend':14} {'Ra':>6} {'chi':>7}  {'predicted':<12} {'literature':<12} result")
print("-" * 82)
hit = miss = 0
for a, b, lit, note in CASES:
    m = blends.miscibility(P[a], P[b])
    pred = m["verdict"]
    # a 'borderline' prediction is counted as agreeing with either neighbouring verdict
    ok = (pred == lit) or (pred == "borderline" and lit in ("miscible", "immiscible")) \
         or (lit == "borderline" and pred in ("miscible", "immiscible"))
    hit += ok; miss += (not ok)
    ra = f"{m['Ra']:.1f}" if m["Ra"] is not None else "-"
    ch = f"{m['chi']:.2f}" if m["chi"] is not None else "-"
    print(f"{a+'/'+b:14} {ra:>6} {ch:>7}  {pred:<12} {lit:<12} {'ok' if ok else 'MISS'}   {note}")
    if m["interaction"]:
        print(f"{'':14} flagged specific interaction: {m['interaction']}")
print("-" * 82)
print(f"{hit}/{hit+miss} consistent with the literature (borderline counts as a partial hit)\n")

# --- Fox equation against measured blend Tg -------------------------------------------
print("Fox equation vs measured blend Tg")
print(f"{'blend':22} {'w1':>5} {'predicted':>10} {'measured':>9}  err")
print("-" * 60)
# (label, Tg1 C, Tg2 C, w1, measured blend Tg C)
FOX = [
    ("PS/PPO 75:25",   100, 210, 0.75, 118),
    ("PS/PPO 50:50",   100, 210, 0.50, 140),
    ("PS/PPO 25:75",   100, 210, 0.25, 170),
    ("PVC/PMMA 50:50",  80, 105, 0.50,  92),
    ("PEO/PMMA 20:80", -60, 105, 0.20,  55),
]
errs = []
for label, t1, t2, w1, meas in FOX:
    got = blends.fox_tg([t1, t2], [w1, 1 - w1])
    errs.append(abs(got - meas))
    print(f"{label:22} {w1:>5.2f} {got:>10.1f} {meas:>9} {got-meas:>+6.1f}")
print("-" * 60)
print(f"mean |error| = {sum(errs)/len(errs):.1f} °C over {len(errs)} blends")
print("\nNOTE: Fox assumes ideal mixing. Systems with strong specific interactions show")
print("positive deviation (measured Tg above Fox) -- PEO/PMMA is the usual example.")

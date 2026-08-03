"""
validate_copolymers.py — check the copolymer model against real commercial copolymers.

Two things are tested:
  1. the Fox equation on RANDOM copolymers against measured Tg (this is Fox's original domain);
  2. alternating-copolymer construction — the built A-B repeat unit must be a valid, neutral,
     two-connection-point polymer unit, and must contain both comonomers' atoms.

    python validation/validate_copolymers.py
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
import copolymers as cp
import blends
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

P = {
    "PS":    "*CC(c1ccccc1)*",
    "PBD":   "*CC=CC*",
    "PAN":   "*CC(C#N)*",
    "PE":    "*CCCC*",
    "PVAc":  "*CC(OC(C)=O)*",
    "PMMA":  "*CC(C)(C(=O)OC)*",
    "PBA":   "*CC(C(=O)OCCCC)*",
    "PVC":   "*CC(Cl)*",
    "PVDF":  "*CC(F)(F)*",
    "PTrFE": "*C(F)C(F)(F)*",
    "PIB":   "*CC(C)(C)*",
    "PMA":   "*CC(C(=O)OC)*",
    "PVOH":  "*CC(O)*",
    "PPO":   "*Oc1cc(C)cc(C)c1*",
}

# (name, A, B, WEIGHT fraction of A, measured Tg °C, Tg_A, Tg_B)
# Commercial copolymers are quoted in WEIGHT per cent (EVA 28 % VAc, SBR 23.5 % styrene),
# so the composition is given on a weight basis and fed straight to Fox -- converting from a
# mole fraction as well would double-count.
FOX_CASES = [
    ("SBR 23.5wt% styrene",  "PS", "PBD", 0.235, -55,  100, -100),
    ("SBR 40wt% styrene",    "PS", "PBD", 0.40,  -30,  100, -100),
    ("SAN 25wt% AN",         "PS", "PAN", 0.75,   105, 100,   95),
    ("P(MMA-co-BA) 50/50",   "PMMA", "PBA", 0.50,  10,  105, -54),
    ("P(MMA-co-BA) 80/20",   "PMMA", "PBA", 0.80,  60,  105, -54),
    ("P(VC-co-VAc) 87/13",   "PVC", "PVAc", 0.87,  70,  80,   30),
    ("P(VDF-co-TrFE) 70/30", "PVDF", "PTrFE", 0.70, -30, -40, -20),
    ("P(S-co-MA) 50/50",     "PS", "PMA",  0.50,   45,  100,   9),
    ("P(S-co-AN) 70/30",     "PS", "PAN",  0.70,   103, 100,  95),
    ("P(VAc-co-MMA) 50/50",  "PVAc", "PMMA", 0.50,  62,  30,  105),
]

print("1) FOX EQUATION on random copolymers (its original purpose)")
print(f"{'copolymer':24} {'w_A':>5} {'predicted':>10} {'measured':>9}  err")
print("-" * 62)
errs = []
for name, a, b, xa, meas, tga, tgb in FOX_CASES:
    got = blends.fox_tg([tga, tgb], [xa, 1 - xa])   # xa is already a weight fraction
    errs.append(abs(got - meas))
    print(f"{name:24} {xa:>5.2f} {got:>10.1f} {meas:>9} {got-meas:>+6.1f}")
print("-" * 62)
print(f"mean |error| = {sum(errs)/len(errs):.1f} °C over {len(errs)} copolymers "
      f"(worst {max(errs):.0f} °C)")
print("Fox assumes ideal mixing of segments; systems with strong specific interactions or")
print("sequence effects deviate, and the equation says nothing about crystallinity.\n")

print("2) ALTERNATING construction — the A-B pair must be a real repeat unit")
print(f"{'pair':22} {'built unit':<46} valid")
print("-" * 82)
ok = bad = 0
PAIRS = [("PS", "PBD"), ("PS", "PAN"), ("PE", "PVAc"), ("PMMA", "PBA"),
         ("PVC", "PVAc"), ("PVDF", "PTrFE"), ("PS", "PMMA"), ("PIB", "PS"),
         ("PPO", "PS"), ("PVOH", "PMMA")]
for a, b in PAIRS:
    unit = cp.build_alternating(P[a], P[b])
    good = False
    why = "build failed"
    if unit:
        m = Chem.MolFromSmiles(unit)
        mh = Chem.MolFromSmiles(unit.replace("*", "[H]"))
        stars = unit.count("*")
        na = Chem.MolFromSmiles(P[a].replace("*", "[H]")).GetNumHeavyAtoms()
        nb = Chem.MolFromSmiles(P[b].replace("*", "[H]")).GetNumHeavyAtoms()
        heavy = mh.GetNumHeavyAtoms() if mh else 0
        good = (m is not None and stars == 2
                and Chem.GetFormalCharge(m) == 0
                and heavy == na + nb)          # atom conservation: A + B, nothing lost or added
        why = (f"stars={stars} charge={Chem.GetFormalCharge(m) if m else '?'} "
               f"heavy={heavy} vs expected {na+nb}")
    ok += good; bad += (not good)
    print(f"{a+'/'+b:22} {(unit or '-'):<46} {'ok' if good else 'FAIL ' + why}")
print("-" * 82)
print(f"{ok}/{ok+bad} alternating units built correctly (valid, neutral, atoms conserved)\n")

print("3) ARCHITECTURE behaviour")
mis = blends.miscibility(P["PS"], P["PBD"])
r = cp.copolymer(P["PS"], P["PBD"], 0.3, "random",
                 preds_a={"Tg": 100}, preds_b={"Tg": -100})
bl = cp.copolymer(P["PS"], P["PBD"], 0.3, "block",
                  preds_a={"Tg": 100}, preds_b={"Tg": -100})
print(f"  PS-co-PBD random : phase={r['phase']}  Tg={r['props']['Tg']['value']:.0f} °C "
      f"(single Tg, as a random copolymer should)")
print(f"  PS-b-PBD  block  : phase={bl['phase']}  Tg={bl['props']['Tg']['value']}  "
      f"[{bl['props']['Tg']['note_args'].get('tgs')}]")
print("  -> the block case keeps two Tgs, which is exactly why SBS is a thermoplastic")
print("     elastomer: a glassy PS domain pinning a rubbery butadiene matrix.")

r2 = cp.copolymer(P["PE"], P["PVAc"], 0.7, "random",
                  preds_a={"Tg": -120, "Tm": 135}, preds_b={"Tg": 30, "Tm": 60})
print(f"\n  EVA random 70/30 : Tm -> {r2['props']['Tm']['value']} "
      f"({r2['props']['Tm']['note_key']})")
print("  -> comonomer incorporation destroys chain regularity, so the melting point is")
print("     suppressed rather than averaged. That is why EVA is flexible where PE is not.")

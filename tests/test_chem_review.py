"""
test_chem_review.py -- the chemist's-review stability checks.

Focus: heteroatom-halogen bonds. A GA run produced a repeat unit with a backbone N-F
(*CCC1=CC(C(F)(F)F)=CC(C(F)(F)OC)=C1N(*)F), which is an N-fluoroamine -- an oxidising /
halogenating reagent, not a polymer -- and the review passed it silently.

The hard part is not detecting N-F, it is detecting it WITHOUT flagging C-F, because C-F is
the whole point of PTFE, PVDF and every low-index fluoropolymer. The false-positive guard
below is therefore the more important half of this file.

    python tests/test_chem_review.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chem_review

HALOGEN_KEYS = ("cr_n_halogen", "cr_o_halogen")

MUST_FLAG = [
    ("backbone N-F (found by the GA)", "*CCC1=CC(C(F)(F)F)=CC(C(F)(F)OC)=C1N(*)F"),
    ("N-chloroamine",                  "*CCN(Cl)CC*"),
    ("N-bromo amide",                  "*CC(=O)N(Br)C*"),
    ("hypochlorite ester (O-Cl)",      "*CC(OCl)C*"),
]

MUST_NOT_FLAG = [
    ("PTFE",                                   "*C(F)(F)C(F)(F)*"),
    ("PVDF",                                   "*CC(F)(F)*"),
    ("PVC",                                    "*CC(Cl)*"),
    ("Nylon-6",                                "*NCCCCCC(=O)*"),
    ("fluorinated polyamide (N and F unbonded)",
     "*NC(=O)C(F)(F)C(F)(F)C(=O)NC1=CC=C(*)C=C1"),
    ("sulfonyl fluoride (Nafion precursor)",   "*CC(F)(S(=O)(=O)F)C(F)(F)*"),
    ("chlorinated aramid",                     "*NC(=O)c1ccc(Cl)c(*)c1"),
    ("fluorinated acrylate copolymer",
     "*CCCC(*)(C)C(=O)OCCC(C1=CC=C(F)C(F)=C1F)C(F)(F)C(F)(F)F"),
]


def flags(smi):
    return [n["key"] for n in chem_review.review(smi, {}) if n.get("key") in HALOGEN_KEYS]


def main():
    bad = 0
    print("heteroatom-halogen bonds MUST be flagged")
    print("-" * 70)
    for name, smi in MUST_FLAG:
        hits = flags(smi)
        bad += not hits
        print(f"  {'ok  ' if hits else 'FAIL'}  {name:<42}{hits}")

    print("\nC-F / C-Cl MUST NOT be flagged")
    print("-" * 70)
    for name, smi in MUST_NOT_FLAG:
        hits = flags(smi)
        bad += bool(hits)
        print(f"  {'ok  ' if not hits else 'FAIL'}  {name:<42}{hits}")

    print("-" * 70)
    if bad:
        raise SystemExit(f"{bad} CHEM-REVIEW CHECK(S) FAILED")
    print("ALL CHEM-REVIEW CHECKS PASSED")


if __name__ == "__main__":
    main()

"""
test_retro.py — regression suite for the rule-based retrosynthesis engine.

Locks in the behaviour that past bugs violated, so a future rule edit cannot silently
break it. Run it after ANY change to retro.py:

    python tests/test_retro.py            # plain, no pytest needed
    pytest tests/test_retro.py            # also works under pytest

Bugs this guards against (each was real):
  * pendant groups mistaken for the backbone linkage (PMMA's side-ester -> "polyester")
  * `_siloxane_monomer` firing on mere Si PRESENCE, so pendant-Si vinyl polymers were
    mis-typed as silicones with bogus silanediol monomers
  * AB monomers (nylon-6, PLA, PCL) failing because the linkage spans the '*' boundary
  * atom-conservation false alarms for monomers that legitimately repeat (PEO -> glycol)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import retro
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

FAILURES = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)
    return cond


def route_of(smi):
    r = retro.retro_decompose(smi)
    return r[0] if r else None


# ---------------------------------------------------------------- canonical polymers
# (smiles, expected type substring, a monomer substring that must appear)
CANONICAL = [
    ("*CC(c1ccccc1)*",                          "Vinyl",         "C=Cc1ccccc1"),
    ("*CC(C)(C(=O)OC)*",                        "Vinyl",         "C=C(C)C(=O)OC"),   # PMMA: NOT polyester
    ("*CC(Cl)*",                                "Vinyl",         "C=CCl"),
    ("*CC(C#N)*",                               "Vinyl",         "C=CC#N"),
    ("*C(F)(F)C(F)(F)*",                        "Vinyl",         "F"),
    ("*OCCOC(=O)c1ccc(C(=O)*)cc1",              "Polyester",     "OCCO"),
    ("*OC(C)C(=O)*",                            "Polyester",     "CC(O)C(=O)O"),     # PLA (AB)
    ("*OCCCCCC(=O)*",                           "Polyester",     "O"),               # PCL (AB)
    ("*NCCCCCC(=O)*",                           "Polyamide",     "NCCCCCC(=O)O"),    # nylon-6 (AB)
    ("*OC(=O)Oc1ccc(C(C)(C)c2ccc(*)cc2)cc1",    "Polycarbonate", "O=C(O)O"),
    ("*CCOCCO*",                                "Polyether",     "OCCO"),
    ("*[Si](C)(C)O*",                           "Polysiloxane",  "[Si]"),
    ("*O[Si](C)(C)*",                           "Polysiloxane",  "[Si]"),
    ("*CC=CC*",                                 "diene",         "C=CC=C"),
    ("*Cc1ccc(C*)cc1",                          "xylylene",      "C=c1ccc(=C)cc1"),  # parylene
]

print("== canonical polymers ==")
for smi, want_type, want_mon in CANONICAL:
    r = route_of(smi)
    if not check(r is not None, f"NO ROUTE for {smi}"):
        print(f"  FAIL  {smi}  -> no route"); continue
    t_ok = want_type.lower() in r["type"].lower()
    m_ok = any(want_mon in m for m in r["monomers"])
    v_ok = r.get("verified", False)
    check(t_ok, f"{smi}: expected type ~{want_type!r}, got {r['type']!r}")
    check(m_ok, f"{smi}: expected a monomer containing {want_mon!r}, got {r['monomers']}")
    check(v_ok, f"{smi}: route not verified ({r.get('verify_reason')})")
    status = "ok  " if (t_ok and m_ok and v_ok) else "FAIL"
    print(f"  {status}  {r['type']:34} {r['monomers']}")


# ------------------------------------------------- pendant groups must NOT set the type
# The backbone is all-carbon in every case -> these are vinyl polymers, whatever hangs off.
print("\n== pendant heteroatom must stay Vinyl (backbone wins) ==")
PENDANT = [
    ("*CC(*)SC",            "pendant thioether"),
    ("*CC(*)C(=O)SC",       "pendant thioester"),
    ("*CC(*)C=NC",          "pendant imine"),
    ("*CC(*)OC",            "pendant ether"),
    ("*CC(*)C(=O)OC",       "pendant ester"),
    ("*CC(*)C(=O)NC",       "pendant amide"),
    ("*CC(*)(C)[Si](C)(C)C", "pendant silicon (was mis-typed as silicone)"),
    ("*CC(*)C[Si](C)(C)OC", "pendant silyl ether"),
]
for smi, label in PENDANT:
    r = route_of(smi)
    ok = r is not None and "Vinyl" in r["type"]
    check(ok, f"{label} ({smi}) should be Vinyl, got {r['type'] if r else 'NO ROUTE'}")
    print(f"  {'ok  ' if ok else 'FAIL'}  {label:44} -> {r['type'] if r else 'NO ROUTE'}")


# ---------------------------------------------------------------- never return a wrong route
print("\n== every returned route must be verified ==")
PROBES = [s for s, _, _ in CANONICAL] + [s for s, _ in PENDANT] + [
    "*C(*)(C)C1=CC=CC=C1", "*CCCCC*", "*CC(C)C*", "*C(*)(CCCC)C(=O)OC",
    "*Cc1ccccc1C*", "*CCSCC*", "*N=Cc1ccc(C=N*)cc1", "*CC=CCCC*",
]
unver = [s for s in PROBES if (r := route_of(s)) and not r.get("verified")]
check(not unver, f"unverified routes returned: {unver}")
print(f"  {'ok  ' if not unver else 'FAIL'}  {len(PROBES)} probes, {len(unver)} unverified")


# ---------------------------------------------------------------- exact vs approximate flag
print("\n== round-trip exact/approx flag ==")
RT = [("*CC(c1ccccc1)*", True), ("*CC(Cl)*", True), ("*Cc1ccc(C*)cc1", True),
      ("*C(*)(C)C1=CC=CC=C1", False),                       # geminal -> nearest vinylidene
      ("*OCCOC(=O)c1ccc(C(=O)*)cc1", None)]                 # step-growth -> n/a
for smi, want in RT:
    r = route_of(smi)
    got = r.get("exact") if r else "no route"
    check(got is want, f"{smi}: expected exact={want}, got {got}")
    print(f"  {'ok  ' if got is want else 'FAIL'}  exact={str(got):5}  {smi}")


# ---------------------------------------------------------------- meta / robustness
print("\n== robustness ==")
check(route_of("*Cc1cccc(C*)c1") is None or "xylylene" not in route_of("*Cc1cccc(C*)c1")["type"],
      "meta-xylylene has no quinodimethane and must not be typed parylene")
for junk in ["", "not_a_smiles", "*", "**", "C", None]:
    try:
        retro.retro_decompose(junk)
    except Exception as e:
        check(False, f"retro_decompose crashed on {junk!r}: {e}")
print("  ok    junk inputs handled, meta-xylylene not parylene")


print("\n" + "=" * 60)
if FAILURES:
    print(f"FAILED ({len(FAILURES)}):")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("ALL RETRO REGRESSION TESTS PASSED")


def test_retro_regression():          # pytest entry point
    assert not FAILURES, FAILURES

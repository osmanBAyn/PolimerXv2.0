"""
chem_review.py — chemist-facing sanity review of a designed polymer.

The property models will happily return a number for every property of every structure. Some
of those numbers are physically meaningless for the polymer in question (a melting point for an
amorphous polymer), and some structures carry groups that would not survive as a stable repeat
unit (a pendant acid next to a pendant alcohol will self-condense). This module flags exactly
those things, so a chemist reading the output sees the caveats before they have to raise them.

    review(smiles, preds) -> [ {level, topic, message}, ... ]
        level: 'error'  the structure or number is not physically meaningful
               'warn'   likely a problem, worth a look
               'info'   context a chemist would want stated

Nothing here is a hard filter: the GA is meant to find novel chemistry, and an unusual motif is
not automatically wrong. These are notes, not rejections.
"""
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


def _capped(smi):
    """
    Repeat unit with each '*' replaced by a METHYL carbon, plus the set of cap indices.

    Capping with [H] (the obvious choice) INVENTS functional groups that the polymer does not
    have: '*OC(C)C(=O)*' (PLA) becomes HO-CH(CH3)-CHO, i.e. a hydroxyl and an aldehyde, so a
    plain polyester would be reported as "carries both -COOH and -OH". Capping with carbon
    keeps the chain-continuation atoms carbon-like and invents nothing; matches that touch a
    cap are additionally ignored.
    """
    try:
        m = Chem.MolFromSmiles(str(smi))
        if m is None:
            return None, set()
        rw = Chem.RWMol(m)
        caps = set()
        for a in rw.GetAtoms():
            if a.GetAtomicNum() == 0:
                a.SetAtomicNum(6)
                a.SetNoImplicit(False)
                a.SetFormalCharge(0)
                caps.add(a.GetIdx())
        mm = rw.GetMol()
        Chem.SanitizeMol(mm)
        return mm, caps
    except Exception:
        return None, set()


def _mol(smi):
    """Backwards-compatible handle: the carbon-capped molecule (never [H]-capped)."""
    return _capped(smi)[0]


def _m(smi):
    """Molecule that KEEPS the '*' so backbone/pendant questions can be asked."""
    try:
        return Chem.MolFromSmiles(str(smi))
    except Exception:
        return None


def _has(mol, smarts):
    p = Chem.MolFromSmarts(smarts)
    return mol is not None and p is not None and mol.HasSubstructMatch(p)


def _count(mol, smarts, caps=frozenset()):
    """Count matches, ignoring any that involve a chain-continuation cap atom."""
    p = Chem.MolFromSmarts(smarts)
    if mol is None or p is None:
        return 0
    return sum(1 for m in mol.GetSubstructMatches(p) if not (set(m) & set(caps)))


# --------------------------------------------------------------------- crystallinity
def likely_amorphous(smi):
    """
    Is this repeat unit likely to be AMORPHOUS (so a melting point is meaningless)?

    Crystallinity needs chain regularity and efficient packing. The reliable structural
    signals we can read are:
      * a bulky/rigid pendant on a saturated backbone (atactic PS, PMMA, PC) -> amorphous;
      * an unsubstituted, regular, linear backbone (PE, POM, PET, PA6) -> semi-crystalline.
    Tacticity, which really decides it for vinyl polymers, is NOT encoded in a SMILES repeat
    unit at all -- so this returns a likelihood, and the caller words it as such.
    """
    mol = _mol(smi)
    star = _m(smi)
    if mol is None or star is None:
        return None
    stars = [a.GetIdx() for a in star.GetAtoms() if a.GetAtomicNum() == 0]
    if len(stars) != 2:
        return None
    # backbone = shortest path between the two connection points
    try:
        nb = [star.GetAtomWithIdx(s).GetNeighbors()[0].GetIdx() for s in stars]
        path = set(Chem.GetShortestPath(star, nb[0], nb[1])) if nb[0] != nb[1] else {nb[0]}
    except Exception:
        return None
    # A ring counts as a bulky PENDANT only if none of its atoms sit on the backbone.
    # A ring the chain runs THROUGH (PET, PEEK) is backbone and does not stop crystallisation.
    ri = star.GetRingInfo()
    pendant_ring = False
    for ring in ri.AtomRings():
        if not (set(ring) & path):                    # ring entirely off the backbone
            pendant_ring = True
            break
    # Backbone carbon carrying >=2 bulky side groups (PMMA's alpha-methyl + ester) hinders
    # packing. Fluorine is deliberately excluded: it is small enough that PTFE and PVDF are
    # highly crystalline.
    def _bulky(nbr):
        return nbr.GetAtomicNum() > 1 and nbr.GetSymbol() != "F"
    quart = any(star.GetAtomWithIdx(i).GetAtomicNum() == 6
                and star.GetAtomWithIdx(i).GetDegree() == 4
                and sum(1 for nbr in star.GetAtomWithIdx(i).GetNeighbors()
                        if nbr.GetIdx() not in path and _bulky(nbr)) >= 2
                for i in path if i < star.GetNumAtoms())
    if pendant_ring or quart:
        return True
    # An aromatic ring IN the backbone still packs well when it is unsubstituted (PEEK, PPS,
    # PET) but not when it carries alkyl groups: PPO's 2,6-dimethyl substitution is precisely
    # why it is amorphous with a Tg and no Tm.
    for ring in ri.AtomRings():
        if not (set(ring) & path):
            continue
        for idx in ring:
            for nbr in star.GetAtomWithIdx(idx).GetNeighbors():
                if (nbr.GetIdx() not in ring and nbr.GetIdx() not in path
                        and nbr.GetAtomicNum() == 6 and not nbr.GetIsAromatic()):
                    return True            # alkyl-substituted backbone arene -> amorphous
    return False


# --------------------------------------------------------------------- structural reviews
def review(smi, preds=None, blend_context=None):
    """Return a list of chemist-facing notes about this repeat unit and its predictions."""
    out = []
    preds = preds or {}
    mol, caps = _capped(smi)
    star = _m(smi)
    if mol is None:
        return [{"level": "error", "key": "cr_bad_smiles", "args": {},
                 "topic": "structure",
                 "message": "SMILES does not parse to a valid molecule"}]

    n_star = str(smi).count("*")
    if n_star < 2:
        out.append({"level": "error", "key": "cr_need_two_stars", "args": {},
                    "topic": "structure",
                    "message": "a repeat unit needs two connection points '*'"})
    elif n_star > 2:
        out.append({"level": "info", "key": "cr_multistar", "args": {"n": n_star},
                    "topic": "architecture",
                    "message": f"{n_star} connection points: this is a branch/crosslink "
                               "junction, not a linear repeat unit. Properties assume a "
                               "linear chain and will shift once the network forms"})

    # --- reactive pendant pairs that would not survive as a stable repeat unit ---
    acid = _count(mol, "C(=O)[OX2H]", caps)
    alco = _count(mol, "[OX2H][CX4]", caps) + _count(mol, "[OX2H]c", caps)
    amine = _count(mol, "[NX3;H1,H2]", caps)
    epox = _count(mol, "C1OC1", caps)
    isoc = _count(mol, "[NX2]=[CX2]=[OX1]", caps)
    if acid and alco:
        out.append({"level": "error", "key": "cr_acid_alcohol", "args": {},
                    "topic": "stability",
                    "message": "carries both -COOH and -OH: these will self-esterify on "
                               "processing, so this is not a stable repeat unit "
                               "(it would branch/crosslink or keep polymerising)"})
    if acid and amine:
        out.append({"level": "error", "key": "cr_acid_amine", "args": {},
                    "topic": "stability",
                    "message": "carries both -COOH and -NH: will self-amidate on heating; "
                               "not a stable repeat unit"})
    if epox and (acid or amine or alco):
        out.append({"level": "warn", "key": "cr_epoxide", "args": {},
                    "topic": "stability",
                    "message": "an epoxide together with a nucleophile (-OH/-NH/-COOH) will "
                               "ring-open and crosslink"})
    if isoc:
        out.append({"level": "warn", "key": "cr_isocyanate", "args": {},
                    "topic": "stability",
                    "message": "free isocyanate is highly reactive and moisture sensitive; "
                               "it would not persist in an isolated polymer"})
    if _count(mol, "[OX2][OX2]", caps):
        out.append({"level": "error", "key": "cr_peroxide", "args": {},
                    "topic": "stability",
                    "message": "peroxide linkage: thermally unstable/explosive"})
    if _count(mol, "[NX3][NX3]", caps) and not _has(mol, "[NX3;R][NX3;R]"):
        out.append({"level": "warn", "key": "cr_nn_bond", "args": {},
                    "topic": "stability",
                    "message": "acyclic N-N bond: usually thermally labile"})

    # --- is a melting point even meaningful? ---
    amorph = likely_amorphous(smi)
    if preds.get("Tm") is not None and amorph:
        out.append({"level": "warn", "key": "cr_tm_amorphous", "args": {"tm": f"{preds['Tm']:.0f}"},
                    "topic": "Tm",
                    "message": f"Tm is reported ({preds['Tm']:.0f} °C) but this structure "
                               "looks AMORPHOUS (bulky pendant / hindered backbone). An "
                               "amorphous polymer has no melting point -- treat Tm as "
                               "meaningless here and use Tg. Note tacticity is not encoded "
                               "in a repeat unit, so this is a structural inference"})
    elif preds.get("Tm") is not None and amorph is False:
        out.append({"level": "info", "key": "cr_tm_plausible", "args": {},
                    "topic": "Tm",
                    "message": "backbone looks regular enough to crystallise, so Tm is "
                               "plausible -- but crystallinity also depends on tacticity and "
                               "thermal history, which a repeat unit cannot express"})

    # --- Tg vs Tm consistency ---
    tg, tm, td = preds.get("Tg"), preds.get("Tm"), preds.get("Td")
    if tg is not None and tm is not None and not amorph and tg >= tm:
        out.append({"level": "warn", "key": "cr_tg_above_tm", "args": {"tg": f"{tg:.0f}", "tm": f"{tm:.0f}"},
                    "topic": "Tg/Tm",
                    "message": f"Tg ({tg:.0f} °C) is not below Tm ({tm:.0f} °C); for a "
                               "semi-crystalline polymer Tg < Tm always holds, so at least "
                               "one of the two predictions is off"})
    # A polymer cannot soften or melt ABOVE the temperature at which it decomposes.
    # The audit found ~4% of GA structures with such a pair, so state it rather than let a
    # chemist find it: it means at least one of the two models is wrong for this structure.
    if tg is not None and td is not None and tg > td:
        out.append({"level": "warn", "topic": "Tg/Td",
                    "key": "cr_tg_above_td", "args": {"tg": f"{tg:.0f}", "td": f"{td:.0f}"},
                    "message": f"Tg ({tg:.0f} °C) is above Td ({td:.0f} °C): a polymer cannot "
                               "soften above the temperature at which it decomposes, so one of "
                               "these two predictions is wrong for this structure"})
    if tm is not None and td is not None and not amorph and tm > td:
        out.append({"level": "warn", "topic": "Tm/Td",
                    "key": "cr_tm_above_td", "args": {"tm": f"{tm:.0f}", "td": f"{td:.0f}"},
                    "message": f"Tm ({tm:.0f} °C) is above Td ({td:.0f} °C): the polymer would "
                               "decompose before melting, so it has no accessible melt window "
                               "(and one of the two predictions is likely wrong)"})

    # empirical Boyer-Beaman rule: Tg/Tm ~ 0.5-0.8 in Kelvin
    if tg is not None and tm is not None and not amorph and tm > -273:
        ratio = (tg + 273.15) / (tm + 273.15)
        if 0 < ratio < 0.4 or ratio > 0.95:
            out.append({"level": "info", "key": "cr_boyer_beaman", "args": {"ratio": f"{ratio:.2f}"},
                        "topic": "Tg/Tm",
                        "message": f"Tg/Tm = {ratio:.2f} in K; the Boyer-Beaman rule puts most "
                                   "polymers at 0.5-0.8, so this pair is unusual"})

    # --- halogen / flame-retardancy context a chemist will ask about ---
    if _count(mol, "[F,Cl,Br,I]", caps) and preds.get("LOI") is not None and preds["LOI"] > 30:
        out.append({"level": "info", "key": "cr_halogen_loi", "args": {},
                    "topic": "LOI",
                    "message": "high LOI here comes from halogen content. Halogenated flame "
                               "retardancy is effective but is being regulated out in many "
                               "markets (and Br/Cl release corrosive HX on combustion)"})

    # --- synthesis realism ---
    if _count(mol, "c", caps) >= 18:
        out.append({"level": "info", "key": "cr_very_aromatic", "args": {},
                    "topic": "synthesis",
                    "message": "heavily aromatic: likely poor solubility and a very high "
                               "processing temperature; melt processing may not be possible"})
    if Descriptors.MolWt(mol) > 500:
        out.append({"level": "info", "key": "cr_big_unit", "args": {"mw": f"{Descriptors.MolWt(mol):.0f}"},
                    "topic": "synthesis",
                    "message": f"large repeat unit ({Descriptors.MolWt(mol):.0f} g/mol): a "
                               "multi-step monomer synthesis is implied"})

    # --- blend context ---
    if blend_context:
        out.append({"level": "info", "topic": "blend", "key": None, "args": {},
                    "message": blend_context})

    return out


def summary(notes):
    """(n_error, n_warn, n_info) for a quick badge."""
    return (sum(1 for n in notes if n["level"] == "error"),
            sum(1 for n in notes if n["level"] == "warn"),
            sum(1 for n in notes if n["level"] == "info"))

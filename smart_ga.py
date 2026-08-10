"""
smart_ga.py
-----------
Heuristic, chemistry-aware genetic operators for POLSEN.

The original GA mutated SELFIES token-by-token (atom-by-atom) and reseeded
randomly on failure. That made the search blind and it frequently destroyed
important motifs (e.g. building a benzene ring and then deleting it).

This module adds three capabilities, all working at the RDKit / SMILES level
so they respect chemistry:

  1. Knowledge-guided SEEDING       -> start from real base polymers that are
                                       known to move the target property in the
                                       right direction (build_seed_population).
  2. FRAGMENT-based mutation        -> attach whole chemically-valid groups or
                                       insert backbone spacers instead of single
                                       atoms (directional_fragment_mutate).
  3. STRUCTURE PRESERVATION         -> removals only touch non-ring, non-functional
                                       leaf atoms; crossover swaps whole acyclic
                                       substituents; a ring-count guard (used in
                                       appv2) never lets a mutation drop a ring.

All operators take/return polymer SMILES using '*' as the two connection points.
Every function is defensive: on any failure it returns None so the GA can simply
keep the parent instead of crashing.
"""

import random
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

import functools
import selfies as sf


# ---------------------------------------------------------------------------
# Base polymer library (repeat unit, '*' = backbone connection points)
# ---------------------------------------------------------------------------
# Small repeat units are written as short segments (>=2 units) so they clear the
# >=4-heavy-atom polymer-validity rule that the GA enforces on every candidate.
BASE_POLYMERS = {
    "PE":   "*CCCC*",                     # polyethylene segment
    "PP":   "*CC(C)CC(C)*",              # polypropylene segment
    "PIB":  "*CC(C)(C)*",                 # polyisobutylene
    "PB":   "*CC=CC*",                    # polybutadiene
    "PDMS": "*[Si](C)(C)O*",             # silicone (very low Tg / high permeability)
    "PEO":  "*CCOCCO*",                   # poly(ethylene oxide) segment
    "PPO":  "*CC(C)O*",                   # poly(propylene oxide)
    "PTMO": "*CCCCO*",                    # poly(tetramethylene oxide)
    "POM":  "*COCOC*",                    # polyoxymethylene segment
    "PVC":  "*CC(Cl)CC(Cl)*",            # poly(vinyl chloride) segment
    "PVDF": "*CC(F)(F)*",
    "PTFE": "*C(F)(F)C(F)(F)*",
    "PS":   "*CC(c1ccccc1)*",            # polystyrene
    "PMMA": "*CC(C)(C(=O)OC)*",
    "PMA":  "*CC(C(=O)OC)*",
    "PAN":  "*CC(C#N)*",                  # polyacrylonitrile
    "PVA":  "*CC(O)CC(O)*",              # poly(vinyl alcohol) segment
    "PVAc": "*CC(OC(C)=O)*",             # poly(vinyl acetate)
    "PET":  "*OCCOC(=O)c1ccc(C(=O)*)cc1",
    "PBT":  "*OCCCCOC(=O)c1ccc(C(=O)*)cc1",
    "PLA":  "*OC(C)C(=O)*",              # polylactic acid
    "PCL":  "*OCCCCCC(=O)*",             # polycaprolactone
    "PGA":  "*OCC(=O)*",                  # polyglycolic acid
    "N6":   "*NCCCCCC(=O)*",             # nylon-6
    "N66":  "*NCCCCCCNC(=O)CCCCC(=O)*",  # nylon-6,6
    "PC":   "*OC(=O)Oc1ccc(C(C)(C)c2ccc(*)cc2)cc1",  # bisphenol-A polycarbonate
    "PSU":  "*Oc1ccc(cc1)S(=O)(=O)c1ccc(*)cc1",       # polysulfone
    "PEEK": "*Oc1ccc(cc1)C(=O)c1ccc(*)cc1",
    "PI":   "*N1C(=O)c2ccc(*)cc2C1=O",   # phthalimide-based polyimide unit
    "PU":   "*OCCOC(=O)Nc1ccc(*)cc1",    # urethane unit
}

DEFAULT_DIVERSE = ["PE", "PS", "PET", "PDMS", "PMMA", "N6", "PC", "PEO", "PP", "PAN"]

# Which base polymers push a property "low" vs "high".
PROPERTY_PROFILES = {
    "Tg":            {"low": ["PDMS", "PEO", "PPO", "PTMO", "PIB", "PE", "PB", "PCL", "POM"],
                      "high": ["PI", "PSU", "PEEK", "PC", "PS", "PAN", "PET", "N66", "PMMA"]},
    "Tm":            {"low": ["PIB", "PMMA", "PMA", "PPO", "PDMS"],
                      "high": ["N6", "N66", "PET", "PBT", "PEEK", "PE", "POM"]},
    "Td":            {"low": ["PVC", "PVAc", "PMMA", "POM", "PE"],
                      "high": ["PI", "PEEK", "PSU", "PET", "PS", "PC"]},
    "EPS":           {"low": ["PTFE", "PE", "PP", "PS", "PIB"],
                      "high": ["PVDF", "PAN", "PVA", "PVAc", "N6"]},
    "BandgapBulk":   {"high": ["PE", "PP", "PDMS", "PTFE", "PEO", "PIB"],
                      "low": ["PB", "PS", "PI", "PEEK", "PET", "PC"]},
    "GasPerma":      {"high": ["PDMS", "PB", "PPO", "PEO", "PIB"],
                      "low": ["PAN", "PET", "N6", "PVA", "PVC"]},
    "Refractive":    {"high": ["PS", "PC", "PSU", "PET", "PEEK", "PVC"],
                      "low": ["PTFE", "PDMS", "PE", "PVDF"]},
    "LOI":           {"high": ["PVC", "PTFE", "PVDF", "PI", "PEEK", "PSU", "PAN"],
                      "low": ["PE", "PP", "PMMA", "POM", "PEO", "PDMS"]},
    "Solubility":    {"high": ["PVA", "PAN", "N6", "PEO", "PET"],
                      "low": ["PE", "PIB", "PDMS", "PTFE", "PP"]},
    "ThermalCond":   {"high": ["PE", "PEEK", "PET"],
                      "low": ["PDMS", "PMMA", "PS"]},
    "CTE":           {"low": ["PI", "PEEK", "PET", "PC", "PS"],
                      "high": ["PE", "PDMS", "PIB", "PEO"]},
    "Recyclability": {"high": ["PET", "PLA", "PCL", "PGA", "N6", "PC"],
                      "low": ["PE", "PP", "PTFE", "PVC", "PS"]},
    "Degradability": {"high": ["PLA", "PCL", "PGA", "PET", "N6"],
                      "low": ["PE", "PP", "PTFE", "PS", "PVC"]},
    "Hansen":        {"high": ["PVA", "PAN", "N6", "PEO", "PET"],
                      "low": ["PE", "PIB", "PDMS", "PTFE", "PP"]},
}
# Band gap chain/crystal share the bulk profile.
PROPERTY_PROFILES["BandgapChain"] = PROPERTY_PROFILES["BandgapBulk"]
PROPERTY_PROFILES["BandgapCrystal"] = PROPERTY_PROFILES["BandgapBulk"]


# ---------------------------------------------------------------------------
# Fragment library (pendant groups, by structural effect)
# ---------------------------------------------------------------------------
F_FLEX  = ["CCCCCC", "CCCC", "CC", "OCCCC", "OCCOCC", "OC"]        # flexible / plasticising
F_RIGID = ["c1ccccc1", "C(C)(C)C", "C1CCCCC1", "c1ccc(C)cc1"]      # bulky / rigid
F_POLAR = ["O", "C(=O)O", "C(=O)N", "C#N", "[N+](=O)[O-]", "C(=O)OC"]  # polar / H-bonding
F_FLUOR = ["F", "C(F)(F)F", "C(F)(F)C(F)(F)F"]                     # fluorinated
F_HALO  = ["Cl", "Br"]
F_AROM  = ["c1ccccc1", "c1ccc2ccccc2c1", "C=Cc1ccccc1"]           # aromatic / conjugated
F_ESTERAMIDE = ["OC(C)=O", "C(=O)N", "O", "C(=O)OC"]              # hydrolysable
F_PHOS  = ["P(=O)(OC)OC"]
F_SULF  = ["Sc1ccccc1", "SC"]

NEUTRAL_FRAGMENTS = ["C", "CC", "O", "c1ccccc1", "F", "C(=O)OC"]

# Pendant-group direction per property.
PROPERTY_FRAGMENTS = {
    "Tg":            {"increase": F_RIGID + F_POLAR,          "decrease": F_FLEX},
    "Tm":            {"increase": F_POLAR + F_RIGID,          "decrease": F_FLEX},
    "Td":            {"increase": F_AROM + ["C(=O)N"],        "decrease": F_FLEX + ["OC"]},
    "EPS":           {"increase": ["C#N", "O", "[N+](=O)[O-]", "C(=O)N"], "decrease": F_FLUOR + ["CCCCCC"]},
    "BandgapBulk":   {"increase": F_FLEX,                     "decrease": F_AROM},
    "GasPerma":      {"increase": ["C(C)(C)C", "[Si](C)(C)C", "C(C)(C)c1ccccc1"] + F_FLEX, "decrease": F_POLAR},
    "Refractive":    {"increase": F_AROM + F_HALO + F_SULF,   "decrease": F_FLUOR + ["CCCCCC"]},
    "LOI":           {"increase": F_HALO + F_FLUOR + F_PHOS + ["c1ccccc1"], "decrease": ["CCCCCC", "CCCC"]},
    "Solubility":    {"increase": ["O", "C(=O)O", "C#N", "C(=O)N"], "decrease": ["CCCCCC"] + F_FLUOR},
    "ThermalCond":   {"increase": F_AROM,                     "decrease": ["C(C)(C)C", "CCCCCC"]},
    "CTE":           {"increase": F_FLEX,                     "decrease": F_RIGID + F_AROM},
    "Recyclability": {"increase": F_ESTERAMIDE,               "decrease": F_AROM + ["CCCCCC"]},
    "Degradability": {"increase": F_ESTERAMIDE,               "decrease": F_AROM + F_HALO + ["CCCCCC"]},
    "Hansen":        {"increase": ["O", "C(=O)O", "C#N", "C(=O)N"], "decrease": ["CCCCCC"] + F_FLUOR},
}
PROPERTY_FRAGMENTS["BandgapChain"] = PROPERTY_FRAGMENTS["BandgapBulk"]
PROPERTY_FRAGMENTS["BandgapCrystal"] = PROPERTY_FRAGMENTS["BandgapBulk"]

# Backbone spacers (inserted into a backbone bond).
S_FLEX  = ["O", "OCC", "CC", "CCCC", "[Si](C)(C)O", "OCCO"]
S_RIGID = ["C(=O)N", "C(=O)O", "C(=O)", "c1ccccc1"]
NEUTRAL_SPACERS = ["CC", "O", "C(=O)"]

PROPERTY_SPACERS = {
    "Tg":            {"increase": S_RIGID,                    "decrease": S_FLEX},
    "Tm":            {"increase": S_RIGID,                    "decrease": S_FLEX},
    "Td":            {"increase": ["c1ccccc1", "C(=O)N"],     "decrease": S_FLEX},
    "CTE":           {"increase": S_FLEX,                     "decrease": S_RIGID},
    "GasPerma":      {"increase": ["[Si](C)(C)O", "O", "CCCC"], "decrease": ["C(=O)N", "C(=O)O"]},
    "Degradability": {"increase": ["C(=O)O", "C(=O)N", "O"],  "decrease": ["CC", "c1ccccc1"]},
    "Recyclability": {"increase": ["C(=O)O", "C(=O)N", "O"],  "decrease": ["CC", "c1ccccc1"]},
}

# Functional groups whose atoms must NOT be stripped by removal ops.
_PROTECT_SMARTS = [Chem.MolFromSmarts(s) for s in [
    "[CX3](=O)[OX2]",     # ester / acid
    "[CX3](=O)[NX3]",     # amide
    "C#N",                # nitrile
    "[N+](=O)[O-]",       # nitro
    "[SX4](=O)(=O)",      # sulfone
    "P(=O)",              # phosphoryl
]]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _mol(smi):
    """Parse a polymer SMILES keeping '*' as dummy atoms."""
    if not smi:
        return None
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


# Motifs that are chemically implausible / unstable in a polymer repeat unit.
# Generated candidates matching any of these are rejected as "not chemically valid".
_UNSTABLE_SMARTS = [Chem.MolFromSmarts(s) for s in [
    "[OX2][OX2]",                 # peroxide  (O-O)
    "[NX3;!R][NX3;!R]",           # hydrazine (acyclic N-N single; ring N-N e.g. pyrazole is fine)
    "[OX2;!R][NX3;!R]",           # acyclic N-O single (ring N-O e.g. isoxazole is fine)
    "[CX4]([OX2H])[OX2H]",        # gem-diol
    "[CX4]([OX2H])[NX3]",         # hemiaminal
    "[CX3](=O)[F,Cl,Br,I]",       # acyl halide
    "[CX3](=O)[OX2][CX3]=O",      # anhydride
    "[CX2]=[CX2]=[CX2]",          # allene / cumulated diene
    "[C-]",                       # carbanion
    "[F,Cl,Br,I][F,Cl,Br,I]",     # halogen-halogen
]]


@functools.lru_cache(maxsize=100000)
def is_chemically_sane(smi):
    """Reject repeat units containing unstable / implausible motifs, or a net charge."""
    try:
        mol = Chem.MolFromSmiles(smi.replace("*", "[H]"))
        if mol is None:
            return False
        # A neutral polymer repeat unit must have NET formal charge 0. This kills GA
        # artifacts like naked alkoxides [O-] or quaternary N+ without a counter-ion
        # (the property models are all trained on neutral polymers), while charge-
        # separated but net-neutral groups -- nitro, N-oxide, azide -- still pass.
        if Chem.GetFormalCharge(mol) != 0:
            return False
        for patt in _UNSTABLE_SMARTS:
            if patt is not None and mol.HasSubstructMatch(patt):
                return False
        return True
    except Exception:
        return False


@functools.lru_cache(maxsize=100000)
def is_valid_polymer_smiles(smi, min_heavy=4, max_heavy=120, sane=True):
    """Chemical validity + polymer requirement (>=2 connection points) + stability."""
    if not smi or smi.count("*") < 2:
        return False
    try:
        mol = Chem.MolFromSmiles(smi.replace("*", "[H]"))
    except Exception:
        return False
    if mol is None:
        return False
    n = mol.GetNumHeavyAtoms()
    if not (min_heavy <= n <= max_heavy):
        return False
    if sane and not is_chemically_sane(smi):
        return False
    return True


@functools.lru_cache(maxsize=100000)
def ring_count(smi):
    """Number of rings (used by the ring-preservation guard)."""
    if not smi:
        return 0
    try:
        m = Chem.MolFromSmiles(smi.replace("*", "[H]"))
        return m.GetRingInfo().NumRings() if m else 0
    except Exception:
        return 0


def smiles_to_selfies(smi):
    """SMILES -> SELFIES using the same '*' <-> [H] convention as appv2."""
    if not smi:
        return None
    try:
        s = sf.encoder(smi.replace("*", "[H]"))
        return s.replace("[H]", "[*]")
    except Exception:
        return None


def get_protected_atoms(mol):
    """Atom indices that must be preserved: ring atoms + functional groups."""
    prot = set()
    try:
        for a in mol.GetAtoms():
            if a.IsInRing():
                prot.add(a.GetIdx())
        for patt in _PROTECT_SMARTS:
            if patt is None:
                continue
            for match in mol.GetSubstructMatches(patt):
                prot.update(match)
    except Exception:
        pass
    return prot


# Functional-group motifs whose COUNT must never drop during a mutation. This is
# how "preserve the important parts" is enforced generally -- not just benzene, but
# every ring, every aromatic system, and these named linkages.
_CONSERVE_GROUPS = [Chem.MolFromSmarts(s) for s in [
    "[CX3](=O)[OX2H0]",              # ester
    "[CX3](=O)[OX2H1]",              # carboxylic acid
    "[CX3](=O)[NX3]",                # amide
    "[NX3][CX3](=O)[OX2]",           # urethane / carbamate
    "[NX3][CX3](=O)[NX3]",           # urea
    "[CX3](=O)[NX3][CX3](=O)",       # imide
    "[OX2][CX3](=O)[OX2]",           # carbonate
    "[SX4](=O)(=O)",                 # sulfone
    "C#N",                           # nitrile
    "[N+](=O)[O-]",                  # nitro
    "P(=O)",                         # phosphoryl
    "[SX2]",                         # thioether / sulfur
]]


def _aromatic_ring_count(mol):
    n = 0
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            n += 1
    return n


@functools.lru_cache(maxsize=100000)   # returned dict is read-only in callers
def motif_profile(smi):
    """
    Count the structural features we consider 'important':
    total rings, aromatic rings, and each conserved functional group.
    Returns a dict of counts, or None on failure.
    """
    try:
        mol = Chem.MolFromSmiles(smi.replace("*", "[H]"))
        if mol is None:
            return None
        prof = {
            "rings": mol.GetRingInfo().NumRings(),
            "arom": _aromatic_ring_count(mol),
        }
        for i, patt in enumerate(_CONSERVE_GROUPS):
            prof["g%d" % i] = len(mol.GetSubstructMatches(patt)) if patt is not None else 0
        return prof
    except Exception:
        return None


def preserves_important(parent_smi, child_smi):
    """
    True only if the child keeps at least as many of every important motif as the
    parent (rings, aromatic rings, and each conserved functional group). Mutations
    may ADD motifs but never remove them -- this is the general structure guard.
    """
    p = motif_profile(parent_smi)
    c = motif_profile(child_smi)
    if p is None or c is None:
        return False
    return all(c.get(k, 0) >= v for k, v in p.items())


# ---------------------------------------------------------------------------
# Fragment-based mutation operators
# ---------------------------------------------------------------------------
def add_pendant_group(smi, frag_smi):
    """Attach a whole chemical group to a carbon that has a free valence."""
    try:
        mol = _mol(smi)
        frag = Chem.MolFromSmiles(frag_smi)
        if mol is None or frag is None or frag.GetNumAtoms() == 0:
            return None

        # Prefer sp3 backbone/side carbons (a real side chain) over substituting a
        # ring carbon; fall back to aromatic carbons for fully aromatic backbones.
        aliph = [a.GetIdx() for a in mol.GetAtoms()
                 if a.GetSymbol() == "C" and not a.GetIsAromatic() and a.GetTotalNumHs() > 0]
        arom = [a.GetIdx() for a in mol.GetAtoms()
                if a.GetSymbol() == "C" and a.GetIsAromatic() and a.GetTotalNumHs() > 0]
        anchors = aliph if (aliph and random.random() < 0.85) else (aliph + arom)
        if not anchors:
            return None
        anchor = random.choice(anchors)

        rw = Chem.RWMol(Chem.CombineMols(mol, frag))
        rw.AddBond(anchor, mol.GetNumAtoms(), Chem.BondType.SINGLE)  # frag atom 0 sits at len(mol)
        new = rw.GetMol()
        Chem.SanitizeMol(new)
        out = Chem.MolToSmiles(new)
        return out if is_valid_polymer_smiles(out) else None
    except Exception:
        return None


def insert_backbone_spacer(smi, spacer_smi):
    """Cut a backbone bond (on the path between the two '*') and splice a spacer in."""
    try:
        mol = _mol(smi)
        spacer = Chem.MolFromSmiles(spacer_smi)
        if mol is None or spacer is None or spacer.GetNumAtoms() == 0:
            return None

        dummies = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
        if len(dummies) < 2:
            return None
        path = Chem.GetShortestPath(mol, dummies[0], dummies[1])
        if not path or len(path) < 2:
            return None

        backbone_bonds = []
        for i in range(len(path) - 1):
            a1, a2 = path[i], path[i + 1]
            b = mol.GetBondBetweenAtoms(a1, a2)
            if (b is not None and b.GetBondType() == Chem.BondType.SINGLE
                    and not b.IsInRing()
                    and mol.GetAtomWithIdx(a1).GetAtomicNum() > 0
                    and mol.GetAtomWithIdx(a2).GetAtomicNum() > 0):
                backbone_bonds.append((a1, a2))
        if not backbone_bonds:
            return None
        a1, a2 = random.choice(backbone_bonds)

        n = mol.GetNumAtoms()
        s_first, s_last = n, n + spacer.GetNumAtoms() - 1
        rw = Chem.RWMol(Chem.CombineMols(mol, spacer))
        rw.RemoveBond(a1, a2)
        rw.AddBond(a1, s_first, Chem.BondType.SINGLE)
        rw.AddBond(a2, s_last, Chem.BondType.SINGLE)
        new = rw.GetMol()
        Chem.SanitizeMol(new)
        out = Chem.MolToSmiles(new)
        return out if is_valid_polymer_smiles(out) else None
    except Exception:
        return None


def remove_leaf_group(smi):
    """Remove a single terminal atom that is NOT in a ring or protected group."""
    try:
        mol = _mol(smi)
        if mol is None:
            return None
        protected = get_protected_atoms(mol)
        cands = [a.GetIdx() for a in mol.GetAtoms()
                 if a.GetDegree() == 1 and a.GetAtomicNum() > 1   # skip dummy(0) and H(1)
                 and not a.IsInRing() and a.GetIdx() not in protected]
        if not cands:
            return None
        rw = Chem.RWMol(mol)
        rw.RemoveAtom(random.choice(cands))
        new = rw.GetMol()
        Chem.SanitizeMol(new)
        out = Chem.MolToSmiles(new)
        return out if is_valid_polymer_smiles(out) else None
    except Exception:
        return None


def _pools_for(prop, want_high):
    """Return (pendant_pool, spacer_pool) pointing in the desired direction."""
    key = "increase" if want_high else "decrease"
    frags = PROPERTY_FRAGMENTS.get(prop, {"increase": F_RIGID, "decrease": F_FLEX})[key]
    spacers = PROPERTY_SPACERS.get(prop, {"increase": S_RIGID, "decrease": S_FLEX}).get(key)
    if not spacers:
        spacers = S_RIGID if want_high else S_FLEX
    return frags, spacers


def directional_fragment_mutate(smi, bias):
    """
    Goal-directed mutation. Picks a property with a strong bias and applies an edit
    (attach group / insert backbone spacer / trim a leaf) that pushes it that way.
    With no strong bias it still adds a chemically valid group (never single atoms).
    """
    strong = [p for p, lvl in (bias or {}).items() if lvl in ("low", "high")]
    if strong:
        prop = random.choice(strong)
        pend_pool, spacer_pool = _pools_for(prop, bias[prop] == "high")
    else:
        pend_pool, spacer_pool = NEUTRAL_FRAGMENTS, NEUTRAL_SPACERS

    roll = random.random()
    if roll < 0.55 and pend_pool:
        return add_pendant_group(smi, random.choice(pend_pool))
    elif roll < 0.85 and spacer_pool:
        return insert_backbone_spacer(smi, random.choice(spacer_pool))
    else:
        return remove_leaf_group(smi)


# ---------------------------------------------------------------------------
# Structure-preserving crossover (swap whole acyclic substituents)
# ---------------------------------------------------------------------------
def _submol_with_open(mol, keep_atoms, open_atom, tag):
    """Extract the sub-molecule spanning keep_atoms, tagging the broken-bond atom."""
    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(open_atom).SetAtomMapNum(tag)
    for idx in sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in keep_atoms),
                      reverse=True):
        rw.RemoveAtom(idx)
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    return m


def _detach_substituent(mol, tag=1):
    """
    Cut one acyclic single bond so the detached fragment contains NO '*'.
    Returns (core_mol, substituent_mol) with the open atoms tagged, or None.
    Rings are never cut, so no important cyclic motif is broken.
    """
    dummy_idxs = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0}
    if not dummy_idxs:
        return None
    bonds = [b for b in mol.GetBonds()
             if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()
             and b.GetBeginAtom().GetAtomicNum() > 0 and b.GetEndAtom().GetAtomicNum() > 0]
    random.shuffle(bonds)
    for b in bonds:
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        rw = Chem.RWMol(mol)
        rw.RemoveBond(a1, a2)
        m = rw.GetMol()
        try:
            Chem.SanitizeMol(m)
            frags = Chem.GetMolFrags(m)
        except Exception:
            continue
        if len(frags) != 2:
            continue
        f0, f1 = set(frags[0]), set(frags[1])
        if dummy_idxs <= f0 and not (dummy_idxs & f1):
            core, sub = f0, f1
        elif dummy_idxs <= f1 and not (dummy_idxs & f0):
            core, sub = f1, f0
        else:
            continue
        if not (1 <= len(sub) <= 30):
            continue
        core_open = a1 if a1 in core else a2
        sub_open = a2 if core_open == a1 else a1
        try:
            return (_submol_with_open(mol, core, core_open, tag),
                    _submol_with_open(mol, sub, sub_open, tag))
        except Exception:
            continue
    return None


def _join_by_tag(core, sub, tag=1):
    """Bond the two tagged open atoms of core + sub back together."""
    combo = Chem.RWMol(Chem.CombineMols(core, sub))
    opens = [a.GetIdx() for a in combo.GetAtoms() if a.GetAtomMapNum() == tag]
    if len(opens) != 2:
        return None
    combo.AddBond(opens[0], opens[1], Chem.BondType.SINGLE)
    for a in combo.GetAtoms():
        a.SetAtomMapNum(0)
    m = combo.GetMol()
    Chem.SanitizeMol(m)
    out = Chem.MolToSmiles(m)
    return out if is_valid_polymer_smiles(out) else None


def smart_crossover(smi1, smi2):
    """
    Exchange whole acyclic substituents between two polymers while keeping each
    backbone (and its two '*') intact. Returns (child1, child2) or None.
    """
    try:
        m1, m2 = _mol(smi1), _mol(smi2)
        if m1 is None or m2 is None:
            return None
        d1 = _detach_substituent(m1)
        d2 = _detach_substituent(m2)
        if not d1 or not d2:
            return None
        core1, sub1 = d1
        core2, sub2 = d2
        child1 = _join_by_tag(core1, sub2)   # backbone 1 + substituent from 2
        child2 = _join_by_tag(core2, sub1)   # backbone 2 + substituent from 1
        if child1 is None and child2 is None:
            return None
        return child1, child2
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Goal analysis + population seeding
# ---------------------------------------------------------------------------
def goal_bias_from_targets(active_props, targets, ranges):
    """
    Turn numeric targets into a coarse direction per property:
    'low' (bottom 40% of range), 'high' (top 40%), or 'mid'.
    """
    bias = {}
    for prop in active_props:
        r = (ranges or {}).get(prop)
        if not r or prop not in (targets or {}):
            bias[prop] = "mid"
            continue
        lo, hi = r["min"], r["max"]
        if hi <= lo:
            bias[prop] = "mid"
            continue
        frac = (targets[prop] - lo) / (hi - lo)
        bias[prop] = "low" if frac < 0.4 else ("high" if frac > 0.6 else "mid")
    return bias


def build_seed_smiles(active_props, bias, n):
    """Collect goal-matched base polymers and diversify them with directional edits."""
    names = set()
    for prop in active_props:
        lvl = bias.get(prop)
        if lvl in ("low", "high"):
            names.update(PROPERTY_PROFILES.get(prop, {}).get(lvl, []))

    # sorted() is load-bearing: `names` is a set of STRINGS, and Python randomises string
    # hashing per process (PYTHONHASHSEED), so iterating it directly gives a different order
    # in every run. That order feeds random.choice(bases) below, which meant the same GA seed
    # produced a different polymer after every app restart -- reproducible within a session,
    # not across restarts. Sorting makes the seed genuinely reproducible without having to
    # disable hash randomisation globally.
    bases = [BASE_POLYMERS[nm] for nm in sorted(names) if nm in BASE_POLYMERS]
    bases = [s for s in bases if is_valid_polymer_smiles(s)]
    if not bases:
        bases = [BASE_POLYMERS[nm] for nm in DEFAULT_DIVERSE if is_valid_polymer_smiles(BASE_POLYMERS[nm])]
    if not bases:
        return []

    out, tries = [], 0
    while len(out) < n and tries < n * 8:
        tries += 1
        s = random.choice(bases)
        if random.random() < 0.6:                       # diversify around the base
            cand = directional_fragment_mutate(s, bias)
            if cand and is_valid_polymer_smiles(cand):
                s = cand
        out.append(s)
    return out


def build_seed_population(active_props, targets, ranges, dataset_selfies,
                          n_seed=60, dataset_keep=240):
    """
    Build the GA's initial population as SELFIES:
      ~1/3 goal-matched base polymers (heuristic starting points)
      + a diverse dataset sample (keeps exploration broad).
    Falls back to the dataset if seeding produces nothing.
    """
    try:
        bias = goal_bias_from_targets(active_props, targets, ranges)
        seed_smis = build_seed_smiles(active_props, bias, n_seed)
        seed_selfies = [x for x in (smiles_to_selfies(s) for s in seed_smis) if x]

        ds = list(dataset_selfies or [])
        random.shuffle(ds)
        pool = list(seed_selfies) + ds[:dataset_keep] + list(seed_selfies)  # seeds weighted twice
        pool = [x for x in pool if x]
        return pool if pool else list(dataset_selfies or [])
    except Exception:
        return list(dataset_selfies or [])

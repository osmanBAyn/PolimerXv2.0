"""
blends.py — polymer BLEND properties and miscibility.

Real materials are almost never neat homopolymers: Noryl is PS/PPO, ABS and HIPS are
rubber-toughened blends, PC/ABS is a commodity engineering alloy. This module takes two or
more repeat units and estimates the blend from the components, using standard relations
rather than a new ML model.

What it does
  miscibility(a, b)        Hansen-distance screening -> miscible / borderline / immiscible
  blend_properties(...)    blend estimate from component properties + composition
  fox_tg(...)              the Fox equation, the workhorse Tg relation for miscible blends

Honest limits (state these to a chemist before they ask):
  * Solubility-parameter matching is a SCREEN, not a proof. It predicts the *enthalpic* term
    only. It systematically mis-calls pairs held together by specific interactions -- PS/PPO
    is miscible through pi-pi stacking despite a large Hansen distance, and PVDF/PMMA through
    a dipole interaction. Those are flagged, not silently mis-reported.
  * Most polymer pairs ARE immiscible: for high molar mass the combinatorial entropy of mixing
    is ~zero, so miscibility is the exception. An "immiscible" verdict is usually right, and is
    often exactly what is wanted (rubber toughening needs a dispersed phase).
  * An immiscible blend's properties depend on MORPHOLOGY (droplet size, adhesion,
    co-continuity) which structure alone cannot predict. For those we report rule-of-mixtures
    bounds, not a single confident number.
"""
import math

R_GAS = 8.314          # J/(mol*K)


# ----------------------------------------------------------------- miscibility screening
def _hansen(smi):
    """(delta_d, delta_p, delta_h, molar_volume) for a repeat unit, or None."""
    try:
        import van_krevelen as vk
        e = vk.vk_estimate(smi)
        hc = e.get("Hansen_components")
        if not hc or e.get("V") in (None, 0):
            return None
        return (hc[0], hc[1], hc[2], e["V"])
    except Exception:
        return None


def hansen_distance(a_smi, b_smi):
    """
    Hansen distance Ra between two repeat units:
        Ra = sqrt( 4*(dd1-dd2)^2 + (dp1-dp2)^2 + (dh1-dh2)^2 )
    The factor 4 on the dispersion term is Hansen's convention. Units MPa^0.5.
    """
    A, B = _hansen(a_smi), _hansen(b_smi)
    if A is None or B is None:
        return None
    return math.sqrt(4 * (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 + (A[2] - B[2]) ** 2)


def flory_huggins_chi(a_smi, b_smi, T=298.15):
    """
    chi from solubility parameters:  chi = V_ref*(d1-d2)^2 / (R*T)  (+0.34 entropic offset).
    V_ref is the smaller repeat-unit molar volume. For high polymers the critical chi is
    ~0, so even a small (d1-d2) drives demixing -- which is why most pairs separate.
    """
    A, B = _hansen(a_smi), _hansen(b_smi)
    if A is None or B is None:
        return None
    d1 = math.sqrt(A[0] ** 2 + A[1] ** 2 + A[2] ** 2)
    d2 = math.sqrt(B[0] ** 2 + B[1] ** 2 + B[2] ** 2)
    v_ref = min(A[3], B[3]) * 1e-6 * 1e6      # cm^3/mol -> keep in cm^3/mol; (MPa^0.5)^2*cm3 = J
    return 0.34 + v_ref * (d1 - d2) ** 2 / (R_GAS * T)


# Pairs whose miscibility is governed by a SPECIFIC interaction that solubility parameters
# cannot see. Detected by functional-group complementarity, and reported as a caveat.
_SPECIFIC = [
    ("aromatic ether + aromatic ring (pi-pi, e.g. PPO/PS)",
     ["[cX3][OX2]", "c1ccccc1"]),
    ("carbonyl + halogenated carbon (dipole/H-bond, e.g. PMMA/PVDF, PMMA/PVC)",
     ["[CX3]=[OX1]", "[CX4][F,Cl]"]),
    ("hydroxyl + carbonyl (H-bond)",
     ["[OX2H]", "[CX3]=[OX1]"]),
]


_INTERACTION_KEY = {
    "aromatic ether + aromatic ring (pi-pi, e.g. PPO/PS)": "blend_inter_pipi",
    "carbonyl + halogenated carbon (dipole/H-bond, e.g. PMMA/PVDF, PMMA/PVC)": "blend_inter_dipole",
    "hydroxyl + carbonyl (H-bond)": "blend_inter_hbond",
}


def _has(smi, smarts):
    """
    Does the repeat unit really contain this group?

    The '*' are capped with CARBON, not [H]: an [H] cap turns '*OC(C)C(=O)*' (PLA) into
    HO-CH(CH3)-CHO, so every polyester and polyether would appear to carry a free hydroxyl
    and be flagged for spurious H-bonding. Matches touching a cap atom are ignored too.
    """
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(str(smi))
        if m is None:
            return False
        rw = Chem.RWMol(m)
        caps = set()
        for a in rw.GetAtoms():
            if a.GetAtomicNum() == 0:
                a.SetAtomicNum(6); a.SetNoImplicit(False); a.SetFormalCharge(0)
                caps.add(a.GetIdx())
        mm = rw.GetMol()
        Chem.SanitizeMol(mm)
        p = Chem.MolFromSmarts(smarts)
        if p is None:
            return False
        return any(not (set(match) & caps) for match in mm.GetSubstructMatches(p))
    except Exception:
        return False


def specific_interaction(a_smi, b_smi):
    """Name a plausible specific interaction between the two units, or None."""
    for label, (g1, g2) in _SPECIFIC:
        if (_has(a_smi, g1) and _has(b_smi, g2)) or (_has(a_smi, g2) and _has(b_smi, g1)):
            return label
    return None


# Thresholds for polymer-POLYMER pairs. Much stricter than the polymer-solvent R0~8 radius,
# because a high-molar-mass pair has essentially no entropy of mixing to help it.
MISC_OK, MISC_EDGE = 3.0, 6.0


def _is_polyolefin(smi):
    """Saturated all-carbon backbone -> a semi-crystalline polyolefin (PE, PP, ...)."""
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(str(smi).replace("*", "[H]"))
        if m is None:
            return False
        if not all(a.GetAtomicNum() in (1, 6) for a in m.GetAtoms()):
            return False                                  # heteroatoms -> not a polyolefin
        if m.HasSubstructMatch(Chem.MolFromSmarts("[cX3]")):
            return False                                  # aromatic -> e.g. PS, not PE/PP
        return not m.HasSubstructMatch(Chem.MolFromSmarts("[CX3]=[CX3]"))   # unsaturated -> rubber
    except Exception:
        return False


def miscibility(a_smi, b_smi, preds_a=None, preds_b=None, tm_crystalline=120.0):
    """
    Screen two repeat units -> dict(Ra, chi, verdict, confidence, interaction, note).
    verdict: 'miscible' | 'borderline' | 'immiscible'

    preds_a/preds_b: optional property dicts. If both components look semi-crystalline
    (high Tm, or a saturated polyolefin backbone) the verdict is downgraded, because two
    crystalline phases do not co-dissolve in the solid state however well their solubility
    parameters match -- PE/PP is the textbook counter-example to a pure delta screen.
    """
    Ra = hansen_distance(a_smi, b_smi)
    if Ra is None:
        return {"Ra": None, "chi": None, "verdict": "unknown", "confidence": "none",
                "interaction": None, "note_key": "blend_misc_note_unknown",
                "caveat_key": None, "interaction_key": None,
                "note": "could not derive solubility parameters for one of the units"}
    chi = flory_huggins_chi(a_smi, b_smi)
    inter = specific_interaction(a_smi, b_smi)
    if Ra <= MISC_OK:
        verdict, conf, note_key = "miscible", "moderate", "blend_misc_note_miscible"
        note = "solubility parameters are close; a single Tg is expected"
    elif Ra <= MISC_EDGE:
        verdict, conf, note_key = "borderline", "low", "blend_misc_note_borderline"
        note = "partial miscibility / weak interphase; expect broadened or two close Tgs"
    else:
        verdict, conf, note_key = "immiscible", "moderate", "blend_misc_note_immiscible"
        note = ("phase-separated; expect two Tgs. This is not necessarily bad -- most "
                "toughened commercial materials rely on a dispersed rubber phase")
    # crystallinity override
    both_olefin = _is_polyolefin(a_smi) and _is_polyolefin(b_smi)
    tma = (preds_a or {}).get("Tm"); tmb = (preds_b or {}).get("Tm")
    both_cryst = (tma is not None and tmb is not None
                  and tma >= tm_crystalline and tmb >= tm_crystalline)
    if (both_olefin or both_cryst) and verdict == "miscible":
        verdict, conf, note_key = "borderline", "low", "blend_misc_note_crystalline"
        note = ("solubility parameters match, BUT both components appear semi-crystalline: "
                "separate crystalline phases normally persist in the solid state regardless "
                "(PE/PP is the classic example). Expect phase separation on cooling")
    caveat_key = None
    if inter and verdict != "miscible":
        conf = "low"
        caveat_key = "blend_misc_caveat_specific"
        note += (f". CAUTION: a specific interaction is possible here ({inter}), which "
                 "solubility parameters cannot capture and which can force miscibility")
    return {"Ra": Ra, "chi": chi, "verdict": verdict, "confidence": conf,
            "interaction": inter, "note": note,
            # translation handles for the UI; 'note'/'interaction' stay as English fallbacks
            "note_key": note_key, "caveat_key": caveat_key,
            "interaction_key": _INTERACTION_KEY.get(inter)}


# ----------------------------------------------------------------- blend property models
def fox_tg(tgs_c, weights):
    """
    Fox equation for a MISCIBLE blend:  1/Tg = sum(w_i / Tg_i), with Tg in KELVIN.
    Returns degrees C. This is the standard first estimate; Gordon-Taylor with a fitted k
    does better but needs a parameter we do not have.
    """
    tot = sum(weights)
    if tot <= 0:
        return None
    inv = 0.0
    for tg, w in zip(tgs_c, weights):
        tk = tg + 273.15
        if tk <= 0:
            return None
        inv += (w / tot) / tk
    return (1.0 / inv) - 273.15 if inv > 0 else None


def _vol_fracs(weights, densities):
    """Weight fractions -> volume fractions (falls back to weight fractions if no density)."""
    if not densities or any(d in (None, 0) for d in densities):
        s = sum(weights)
        return [w / s for w in weights] if s else None
    v = [w / d for w, d in zip(weights, densities)]
    s = sum(v)
    return [x / s for x in v] if s else None


def blend_properties(components, weights, T=298.15):
    """
    Estimate blend properties.
      components : [{'smiles':..., 'preds':{prop: value}}, ...]
      weights    : weight fractions (need not sum to 1)
    Returns {'miscibility':..., 'props':{prop:{'value','rule','note'}}, 'volume_fractions':...}

    Rules used (all standard, none invented):
      Tg          Fox equation if miscible; otherwise the component Tgs are reported separately
      density     additive specific volume (1/rho = sum w_i/rho_i)
      Refractive  Lorentz-Lorenz, volume-fraction additive
      GasPerma    semi-log additive (ln P = sum phi_i ln P_i) for miscible blends;
                  Maxwell dispersed-phase model when phase separated
      Solubility/Hansen  volume-fraction additive
      LOI         weight additive (FLAGGED: flame retardancy is frequently non-additive)
      Td          conservative -- onset is set by the LEAST stable component
      Tm          a crystalline phase keeps its own Tm; not averaged
      others      rule-of-mixtures value plus the series/parallel bounds
    """
    n = len(components)
    if n < 2 or len(weights) != n:
        return None
    s = float(sum(weights))
    if s <= 0:
        return None
    w = [x / s for x in weights]

    dens = []
    for c in components:
        try:
            import van_krevelen as vk
            dens.append(vk.vk_estimate(c["smiles"]).get("density"))
        except Exception:
            dens.append(None)
    phi = _vol_fracs(w, dens) or w

    mis = miscibility(components[0]["smiles"], components[1]["smiles"]) if n == 2 else \
        {"verdict": "unknown", "confidence": "none", "Ra": None, "chi": None,
         "interaction": None, "note": "pairwise screening is only defined for 2 components"}
    miscible = mis["verdict"] == "miscible"

    def vals(prop):
        out = []
        for c in components:
            v = (c.get("preds") or {}).get(prop)
            out.append(None if v is None else float(v))
        return out

    props = {}

    tgs = vals("Tg")
    if all(v is not None for v in tgs):
        if miscible:
            props["Tg"] = {"value": fox_tg(tgs, w), "rule": "Fox equation (miscible)",
                           "rule_key": "blend_rule_fox", "note_key": "blend_note_singletg",
                           "note_args": {}, "note": "single Tg expected"}
        else:
            _tgtxt = ", ".join(f"{v:.0f} °C" for v in tgs)
            props["Tg"] = {"value": None, "rule": "phase separated",
                           "rule_key": "blend_rule_phasesep", "note_key": "blend_note_twotg",
                           "note_args": {"tgs": _tgtxt},
                           "note": "two Tgs retained: " + _tgtxt}

    d = [x for x in dens if x]
    if len(d) == n:
        props["density"] = {"value": 1.0 / sum(wi / di for wi, di in zip(w, dens)),
                            "rule": "additive specific volume",
                            "rule_key": "blend_rule_specvol", "note_key": None,
                            "note_args": {}, "note": ""}

    ri = vals("Refractive")
    if all(v is not None for v in ri):
        # Lorentz-Lorenz mixing on the (n^2-1)/(n^2+2) function, volume-fraction weighted
        ll = sum(p * ((v * v - 1) / (v * v + 2)) for p, v in zip(phi, ri))
        if 0 < ll < 1:
            props["Refractive"] = {"value": math.sqrt((1 + 2 * ll) / (1 - ll)),
                                   "rule": "Lorentz-Lorenz (volume fraction)",
                                   "rule_key": "blend_rule_ll",
                                   "note_key": None if miscible else "blend_note_hazy",
                                   "note_args": {},
                                   "note": "" if miscible else
                                   "phase-separated blends scatter light and are usually hazy"}

    gp = vals("GasPerma")
    if all(v is not None and v > 0 for v in gp):
        if miscible:
            props["GasPerma"] = {"value": math.exp(sum(p * math.log(v) for p, v in zip(phi, gp))),
                                 "rule": "semi-log additive (miscible)",
                                 "rule_key": "blend_rule_semilog", "note_key": None,
                                 "note_args": {}, "note": ""}
        elif n == 2:
            # Maxwell: continuous phase = the majority component, dispersed = the other
            ci = 0 if phi[0] >= phi[1] else 1
            di = 1 - ci
            Pc, Pd, pd = gp[ci], gp[di], phi[di]
            val = Pc * (Pd + 2 * Pc - 2 * pd * (Pc - Pd)) / (Pd + 2 * Pc + pd * (Pc - Pd))
            props["GasPerma"] = {"value": val, "rule": "Maxwell model (dispersed phase)",
                                 "rule_key": "blend_rule_maxwell",
                                 "note_key": "blend_note_continuous",
                                 "note_args": {"n": ci + 1},
                                 "note": f"continuous phase = component {ci + 1}"}

    for prop, rule in (("Solubility", "volume-fraction additive"),
                       ("Hansen", "volume-fraction additive")):
        v = vals(prop)
        if all(x is not None for x in v):
            props[prop] = {"value": sum(p * x for p, x in zip(phi, v)), "rule": rule,
                           "rule_key": "blend_rule_volfrac", "note_key": None,
                           "note_args": {}, "note": ""}

    loi = vals("LOI")
    if all(v is not None for v in loi):
        props["LOI"] = {"value": sum(wi * v for wi, v in zip(w, loi)),
                        "rule": "weight additive",
                        "rule_key": "blend_rule_wtadd", "note_key": "blend_note_loi",
                        "note_args": {},
                        "note": "CAUTION: flame retardancy is often non-additive "
                                "(synergism/antagonism); treat as indicative only"}

    td = vals("Td")
    if all(v is not None for v in td):
        props["Td"] = {"value": min(td), "rule": "least-stable component",
                       "rule_key": "blend_rule_weakest", "note_key": "blend_note_td",
                       "note_args": {},
                       "note": "decomposition onset is governed by the weakest component"}

    tm = vals("Tm")
    if all(v is not None for v in tm):
        _tmtxt = ", ".join(f"{v:.0f} °C" for v in tm)
        props["Tm"] = {"value": None, "rule": "not averaged",
                       "rule_key": "blend_rule_notavg", "note_key": "blend_note_tm_each",
                       "note_args": {"tms": _tmtxt},
                       "note": "each crystalline phase keeps its own Tm: " + _tmtxt}

    for prop in ("ThermalCond", "EPS", "CTE"):
        v = vals(prop)
        if all(x is not None for x in v):
            par = sum(p * x for p, x in zip(phi, v))                      # parallel bound
            ser = 1.0 / sum(p / x for p, x in zip(phi, v)) if all(x for x in v) else None
            props[prop] = {"value": par, "rule": "rule of mixtures (parallel bound)",
                           "rule_key": "blend_rule_rom",
                           "note_key": "blend_note_bounds" if ser else None,
                           "note_args": {"ser": f"{ser:.3g}"} if ser else {},
                           "note": (f"series bound {ser:.3g}; the true value lies between "
                                    "the bounds and depends on morphology") if ser else ""}

    return {"miscibility": mis, "props": props,
            "weight_fractions": w, "volume_fractions": phi, "densities": dens}

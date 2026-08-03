"""
copolymers.py — copolymer properties (random / alternating / block).

Copolymers are arguably even more common than blends — ABS, SBR, EVA, SAN, NBR, P(VDF-TrFE),
EPDM — and they are more tractable, because the two units are covalently bonded: there is no
miscibility question at the segment level, so composition-weighted relations apply directly.

Three architectures, three genuinely different physics:

  alternating   ...ABABAB...  The pair IS a new repeat unit. We build it explicitly and hand it
                to the property models, so this is an EXACT prediction, not a mixing estimate.
  random        ...AABABBA...  One phase, one Tg. Tg from the Fox equation (this is the classic
                use of Fox — it was derived for random copolymers before being borrowed for
                miscible blends); other properties composition-weighted.
  block         ...AAAA-BBBB... Long blocks MICROPHASE-SEPARATE when the segments are
                incompatible, giving TWO Tgs and a domain morphology (this is what makes SBS
                a thermoplastic elastomer). We reuse the blend miscibility screen to decide.

Composition can be given on a mole or a WEIGHT basis (`basis=`). Commercial grades are almost
always quoted in weight per cent -- EVA 28 % VAc, SBR 23.5 % styrene -- and the mixing relations
themselves take weight fractions, so passing basis='weight' avoids a double conversion.
"""
import math
from rdkit import Chem
from rdkit.Chem import Descriptors, RDConfig  # noqa: F401
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import blends


# --------------------------------------------------------------- alternating construction
def build_alternating(a_smi, b_smi):
    """
    Join two repeat units head-to-tail into the single A-B repeat unit of the alternating
    copolymer, keeping one '*' at each end. Returns SMILES, or None if either unit is not a
    well-formed two-connection-point repeat unit.
    """
    A, B = Chem.MolFromSmiles(str(a_smi)), Chem.MolFromSmiles(str(b_smi))
    if A is None or B is None:
        return None
    sa = [x.GetIdx() for x in A.GetAtoms() if x.GetAtomicNum() == 0]
    sb = [x.GetIdx() for x in B.GetAtoms() if x.GetAtomicNum() == 0]
    if len(sa) != 2 or len(sb) != 2:
        return None
    try:
        a_tail_nbr = A.GetAtomWithIdx(sa[1]).GetNeighbors()[0].GetIdx()
        b_head_nbr = B.GetAtomWithIdx(sb[0]).GetNeighbors()[0].GetIdx()
    except IndexError:
        return None
    na = A.GetNumAtoms()
    combo = Chem.RWMol(A)
    combo.InsertMol(B)
    # bond A's tail to B's head, then delete the two '*' that were consumed
    combo.AddBond(a_tail_nbr, na + b_head_nbr, Chem.BondType.SINGLE)
    for idx in sorted([sa[1], na + sb[0]], reverse=True):
        combo.RemoveAtom(idx)
    m = combo.GetMol()
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    out = Chem.MolToSmiles(m)
    return out if out.count("*") == 2 else None


def _unit_mass(smi):
    """Molar mass of a repeat unit (the two '*' contribute nothing)."""
    m = Chem.MolFromSmiles(str(smi).replace("*", "[H]"))
    if m is None:
        return None
    return Descriptors.MolWt(m) - 2 * 1.008


def mole_to_weight(a_smi, b_smi, x_a):
    """Mole fraction of A -> weight fraction of A."""
    ma, mb = _unit_mass(a_smi), _unit_mass(b_smi)
    if not ma or not mb:
        return x_a
    return (x_a * ma) / (x_a * ma + (1 - x_a) * mb)


# --------------------------------------------------------------- the model
ARCHITECTURES = ("random", "alternating", "block")


def copolymer(a_smi, b_smi, x_a=0.5, architecture="random",
              preds_a=None, preds_b=None, predict_fn=None, basis="mole"):
    """
    Estimate a copolymer.

      a_smi, b_smi : the two comonomer repeat units ('*...*')
      x_a          : fraction of A (0-1), interpreted per `basis`
      basis        : 'mole' (default) or 'weight'. Commercial copolymer grades are usually
                     quoted in WEIGHT per cent (EVA 28 % VAc, SBR 23.5 % styrene), so pass
                     basis='weight' to avoid a double conversion.
      architecture : 'random' | 'alternating' | 'block'
      preds_a/b    : property dicts for the two homopolymers
      predict_fn   : callable(smiles) -> preds, used by the 'alternating' route to predict the
                     constructed A-B unit directly

    Returns {'architecture', 'x_a', 'w_a', 'unit' (alternating only), 'phase', 'props', 'notes'}
    where each prop is {'value', 'rule_key', 'note_key', 'note_args'}.
    """
    if architecture not in ARCHITECTURES:
        return None
    w_a = x_a if basis == "weight" else mole_to_weight(a_smi, b_smi, x_a)
    out = {"architecture": architecture, "x_a": x_a, "w_a": w_a,
           "unit": None, "phase": None, "props": {}, "notes": []}

    # ---- alternating: the pair is a real repeat unit, so predict it directly -------------
    if architecture == "alternating":
        unit = build_alternating(a_smi, b_smi)
        out["unit"] = unit
        out["phase"] = "single"
        if unit and predict_fn:
            pr = predict_fn(unit) or {}
            for p, v in pr.items():
                out["props"][p] = {"value": v, "rule_key": "copo_rule_direct",
                                   "note_key": None, "note_args": {}}
            out["notes"].append("copo_note_alternating_exact")
        elif not unit:
            out["notes"].append("copo_note_build_failed")
        return out

    # ---- block: the segments microphase-separate when they would not mix ----------------
    if architecture == "block":
        mis = blends.miscibility(a_smi, b_smi, preds_a, preds_b)
        separated = mis["verdict"] != "miscible"
        out["phase"] = "microphase-separated" if separated else "single"
        out["miscibility"] = mis
        out["notes"].append("copo_note_block_sep" if separated else "copo_note_block_single")

    # ---- random (and miscible block): one phase, composition-weighted --------------------
    if architecture == "random":
        out["phase"] = "single"      # covalently forced to mix at the segment level
    pa, pb = preds_a or {}, preds_b or {}
    w = [w_a, 1 - w_a]

    def both(p):
        va, vb = pa.get(p), pb.get(p)
        return (None if va is None or vb is None else (float(va), float(vb)))

    tg = both("Tg")
    if tg:
        if architecture == "block" and out["phase"] == "microphase-separated":
            out["props"]["Tg"] = {"value": None, "rule_key": "copo_rule_twophase",
                                  "note_key": "copo_note_two_tg",
                                  "note_args": {"tgs": f"{tg[0]:.0f} °C, {tg[1]:.0f} °C"}}
        else:
            out["props"]["Tg"] = {"value": blends.fox_tg(list(tg), w),
                                  "rule_key": "copo_rule_fox", "note_key": None,
                                  "note_args": {}}

    # weight-additive properties (one phase, so a simple mixing rule is the standard estimate)
    for p in ("LOI", "Solubility", "Hansen", "ThermalCond", "EPS", "CTE", "Degradability",
              "Refractive"):
        v = both(p)
        if v:
            out["props"][p] = {"value": w[0] * v[0] + w[1] * v[1],
                               "rule_key": "copo_rule_wtadd", "note_key": None,
                               "note_args": {}}

    gp = both("GasPerma")
    if gp and gp[0] > 0 and gp[1] > 0:
        out["props"]["GasPerma"] = {
            "value": math.exp(w[0] * math.log(gp[0]) + w[1] * math.log(gp[1])),
            "rule_key": "copo_rule_semilog", "note_key": None, "note_args": {}}

    td = both("Td")
    if td:
        out["props"]["Td"] = {"value": min(td), "rule_key": "copo_rule_weakest",
                              "note_key": "copo_note_td", "note_args": {}}

    # Crystallinity is DESTROYED by comonomer incorporation: a random copolymer loses the
    # chain regularity that crystallisation needs, which is exactly why EVA and random
    # ethylene copolymers are flexible where PE is not. So Tm is not simply averaged.
    tm = both("Tm")
    if tm:
        if architecture == "random" and 0.1 < x_a < 0.9:
            out["props"]["Tm"] = {"value": None, "rule_key": "copo_rule_nocryst",
                                  "note_key": "copo_note_tm_suppressed", "note_args": {}}
        else:
            out["props"]["Tm"] = {"value": None, "rule_key": "copo_rule_twophase",
                                  "note_key": "copo_note_tm_each",
                                  "note_args": {"tms": f"{tm[0]:.0f} °C, {tm[1]:.0f} °C"}}
    return out

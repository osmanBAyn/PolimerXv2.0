"""
polymer_md.py — OFFLINE molecular-dynamics property estimation for a polymer repeat unit.

This is the rigorous (and expensive) route for the BULK properties no cheap method verifies
well per-polymer: density, solubility parameter, and — via a temperature scan — Tg and CTE.
It is NOT part of the app and NOT a per-result button: a density run is minutes on a GPU, a
Tg scan is 1-3 hours. Run it offline on a shortlisted candidate.

Engine: OpenMM (GPU via CUDA/OpenCL, CPU fallback). Force field: OpenFF Sage (SMIRNOFF).
Cell: several short H-capped oligomer chains packed into a periodic box, then NPT-compressed.

Pipeline
  make_chain(repeat_smiles, n)      RDKit oligomer of n repeat units, H-capped   (no MD deps)
  build_system(...)                 pack chains -> parametrize -> OpenMM System
  measure_density(...)              minimize -> NVT -> NPT -> average density (g/cm^3)
  solubility_parameter(...)         cohesive energy density -> delta (MPa^0.5)
  tg_scan(..., temps)               density(T) ladder -> Tg (deg C) and CTE (ppm/K)

Usage
  conda install -n polsen -c conda-forge openmm openff-toolkit openmmforcefields
  python md/polymer_md.py "*CC*" --mode density   --units 10 --chains 12
  python md/polymer_md.py "*CC*" --mode tg         --units 12 --chains 12

Accuracy scales with system size and run length (both set small here so it FINISHES). Treat
the defaults as a screening estimate; lengthen `--ns`, `--units`, `--chains` for publication.
"""
import sys, os, argparse
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")


# --------------------------------------------------------------------- chain builder (RDKit only)
def make_chain(repeat_smiles, n):
    """Join n copies of a '*...*' repeat unit head-to-tail; cap the two ends with H. -> RDKit Mol."""
    unit = Chem.MolFromSmiles(str(repeat_smiles))
    if unit is None:
        raise ValueError(f"bad SMILES: {repeat_smiles}")
    stars = [a.GetIdx() for a in unit.GetAtoms() if a.GetAtomicNum() == 0]
    if len(stars) != 2:
        raise ValueError("repeat unit needs exactly two '*' endpoints")
    nb = []
    for s in stars:
        ns = unit.GetAtomWithIdx(s).GetNeighbors()
        if not ns:
            raise ValueError("dangling '*'")
        nb.append(ns[0].GetIdx())
    (head_star, tail_star), (head_nb, tail_nb) = stars, (nb[0], nb[1])
    na = unit.GetNumAtoms()
    combo = Chem.RWMol(unit)
    for _ in range(n - 1):
        combo.InsertMol(unit)
    rm = []
    for i in range(n):
        off = i * na
        if i < n - 1:
            combo.AddBond(off + tail_nb, (i + 1) * na + head_nb, Chem.BondType.SINGLE)
            rm += [off + tail_star, (i + 1) * na + head_star]
    rm += [head_star, (n - 1) * na + tail_star]
    for idx in sorted(set(rm), reverse=True):
        combo.RemoveAtom(idx)
    m = combo.GetMol()
    Chem.SanitizeMol(m)
    m = Chem.AddHs(m)
    return m


def repeat_mass(repeat_smiles):
    """Molar mass of one repeat unit (for density sanity), stars removed."""
    from rdkit.Chem import Descriptors
    m = Chem.MolFromSmiles(str(repeat_smiles).replace('*', '[H]'))
    return Descriptors.MolWt(m) - 2 * 1.008 if m else None


# --------------------------------------------------------------------- MD (OpenMM + OpenFF)
def _lazy_md():
    """Import the heavy MD stack only when actually running a simulation."""
    from openff.toolkit import Molecule, ForceField, Topology
    import openmm, openmm.app as app, openmm.unit as u
    return Molecule, ForceField, Topology, openmm, app, u


def _assign_charges(mol):
    """AM1-BCC if a backend is available, else Gasteiger (always works, lower quality)."""
    try:
        mol.assign_partial_charges("am1bcc")
        return "am1bcc"
    except Exception:
        mol.assign_partial_charges("gasteiger")
        return "gasteiger"


def _rand_rot(rng):
    """Uniform random rotation matrix from a random unit quaternion (numpy only, no scipy)."""
    import numpy as np
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _pack(chain_off, n_chains, density_gcc, seed=42):
    """
    packmol is not on conda-forge for Windows, so pack the cell ourselves: drop n_chains
    randomly-rotated copies of the chain onto a cubic grid in a LOW-density box (no overlaps),
    then NPT compresses it to the real density during equilibration. Returns an openff Topology.
    """
    import numpy as np, copy
    from openff.units import unit as off_unit
    from openff.toolkit import Topology
    rng = np.random.default_rng(seed)
    conf = chain_off.conformers[0].to("angstrom").magnitude       # (N,3) angstrom
    conf = conf - conf.mean(axis=0)                               # center at origin
    mass_dalton = sum(a.mass.m_as(off_unit.dalton) for a in chain_off.atoms)
    total_mass_g = n_chains * mass_dalton * 1.66054e-24
    box_A = (total_mass_g / density_gcc * 1e24) ** (1.0 / 3.0)    # cube side for target density
    g = int(np.ceil(n_chains ** (1.0 / 3.0)))                    # grid points per side
    spacing = box_A / g
    mols, placed = [], 0
    for ix in range(g):
        for iy in range(g):
            for iz in range(g):
                if placed >= n_chains:
                    break
                center = (np.array([ix, iy, iz], float) + 0.5) * spacing
                pos = conf @ _rand_rot(rng).T + center
                m = copy.deepcopy(chain_off)
                m._conformers = [pos * off_unit.angstrom]
                mols.append(m); placed += 1
    top = Topology.from_molecules(mols)
    top.box_vectors = (np.eye(3) * box_A) * off_unit.angstrom
    return top


def build_system(repeat_smiles, n_units, n_chains, ff_name="openff-2.1.0.offxml",
                 build_density=0.15, verbose=True):
    """Return (openmm_system, openmm_topology, positions[nm], off_topology, charge_method)."""
    Molecule, ForceField, Topology, openmm, app, u = _lazy_md()
    # Auto-size: NPT compresses the box toward ~1 g/cm^3; it must stay > 2 x nonbonded cutoff
    # (~0.8 nm here). Guarantee the *compressed* cube is >= ~2.6 nm by adding chains as needed.
    rmass = repeat_mass(repeat_smiles) or 50.0
    min_mass = 602.2 * (2.6 ** 3)                    # daltons for a 2.6 nm cube at 1.0 g/cm^3
    need = int((min_mass / (n_units * rmass)) + 0.999)
    if need > n_chains:
        if verbose:
            print(f"[build] bumping chains {n_chains} -> {need} so the compressed box stays large enough", flush=True)
        n_chains = need
    rdmol = make_chain(repeat_smiles, n_units)
    if AllChem.EmbedMolecule(rdmol, randomSeed=42) != 0:
        if AllChem.EmbedMolecule(rdmol, randomSeed=1, useRandomCoords=True) != 0:
            raise RuntimeError("3D embedding of the chain failed (too rigid / too long)")
    AllChem.MMFFOptimizeMolecule(rdmol, maxIters=1000)
    chain = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)
    cm = _assign_charges(chain)
    if verbose:
        print(f"[build] {n_units}-mer x {n_chains} chains, charges={cm}", flush=True)
    off_top = _pack(chain, n_chains, build_density)
    ff = ForceField(ff_name)
    inter = ff.create_interchange(off_top, charge_from_molecules=[chain])
    system = inter.to_openmm()
    # trim nonbonded cutoff (and its switching distance) for box-size headroom during compression
    for f in system.getForces():
        if isinstance(f, openmm.NonbondedForce):
            try:
                f.setCutoffDistance(0.8 * u.nanometer)
                if f.getUseSwitchingFunction():
                    f.setSwitchingDistance(0.7 * u.nanometer)
            except Exception:
                pass
    omm_top = inter.to_openmm_topology()
    positions = inter.positions.to("nanometer").magnitude
    return system, omm_top, positions, off_top, cm


def _platform(openmm):
    for name in ("CUDA", "OpenCL", "CPU"):
        try:
            p = openmm.Platform.getPlatformByName(name)
            return p, name
        except Exception:
            continue
    return None, "Reference"


def _npt_sim(system, omm_top, positions, T, openmm, app, u, friction=1.0, ts_fs=2.0, pressure_atm=1.0):
    system.addForce(openmm.MonteCarloBarostat(pressure_atm * u.atmosphere, T * u.kelvin, 25))
    integ = openmm.LangevinMiddleIntegrator(T * u.kelvin, friction / u.picosecond, ts_fs * u.femtoseconds)
    plat, pname = _platform(openmm)
    sim = app.Simulation(omm_top, system, integ, plat)
    sim.context.setPositions(positions * u.nanometer if not hasattr(positions, "unit") else positions)
    return sim, pname


def measure_density(repeat_smiles, n_units=10, n_chains=12, T=300.0, ns=0.2, equil_ns=0.4, verbose=True):
    """Minimize -> NVT heat -> NPT; return mean density (g/cm^3) over production.
    equil_ns must be long enough to COMPRESS from the low-density pack; 0.1 ns under-fills."""
    import numpy as np
    Molecule, ForceField, Topology, openmm, app, u = _lazy_md()
    system, omm_top, pos, off_top, cm = build_system(repeat_smiles, n_units, n_chains, verbose=verbose)
    sim, pname = _npt_sim(system, omm_top, pos, T, openmm, app, u)
    if verbose:
        print(f"[md] platform={pname}; minimizing...", flush=True)
    sim.minimizeEnergy(maxIterations=2000)
    sim.context.setVelocitiesToTemperature(T * u.kelvin)
    steps_per_ns = int(1e6 / 2.0)                       # 2 fs timestep
    sim.step(int(equil_ns * steps_per_ns))              # NPT equilibration (compress)
    n_prod = int(ns * steps_per_ns)
    sample_every = max(1, n_prod // 200)
    dens = []
    mol_mass = sum(a.element.mass.value_in_unit(u.dalton) for a in omm_top.atoms())
    for _ in range(0, n_prod, sample_every):
        sim.step(sample_every)
        box = sim.context.getState().getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer)
        vol_nm3 = abs(np.linalg.det(box))
        # density = mass(g) / volume(cm^3); 1 nm^3 = 1e-21 cm^3; mass_dalton*1.6605e-24 g
        dens.append(mol_mass * 1.66054e-24 / (vol_nm3 * 1e-21))
    d = float(np.mean(dens[len(dens) // 3:]))           # drop first third
    if verbose:
        print(f"[result] density(T={T:.0f}K) = {d:.3f} g/cm^3  (+-{np.std(dens[len(dens)//3:]):.3f})", flush=True)
    return d


def tg_scan(repeat_smiles, temps=(400, 375, 350, 325, 300, 275, 250, 225, 200), **kw):
    """Density vs T; Tg = knee of specific-volume(T); CTE from the melt-side slope."""
    import numpy as np
    rows = []
    for T in temps:
        d = measure_density(repeat_smiles, T=float(T), **kw)
        rows.append((float(T), d))
        print(f"  T={T}K  rho={d:.3f}", flush=True)
    T = np.array([r[0] for r in rows]); rho = np.array([r[1] for r in rows])
    sv = 1.0 / rho                                       # specific volume
    # two-line fit: split at each interior point, pick split minimizing total residual
    best = None
    for k in range(2, len(T) - 2):
        p1 = np.polyfit(T[:k], sv[:k], 1); p2 = np.polyfit(T[k:], sv[k:], 1)
        r = (np.sum((np.polyval(p1, T[:k]) - sv[:k]) ** 2) +
             np.sum((np.polyval(p2, T[k:]) - sv[k:]) ** 2))
        if best is None or r < best[0]:
            tg = (p2[1] - p1[1]) / (p1[0] - p2[0])       # intersection
            cte = p1[0] / np.polyval(p1, tg) * 1e6       # melt-side volumetric CTE, ppm/K
            best = (r, tg, cte, p1, p2)
    _, tg, cte, _, _ = best
    print(f"\n[result] Tg = {tg-273.15:.0f} degC   volumetric CTE(melt) = {cte:.0f} ppm/K", flush=True)
    return tg - 273.15, cte, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("smiles")
    ap.add_argument("--mode", choices=["density", "solubility", "tg"], default="density")
    ap.add_argument("--units", type=int, default=10)
    ap.add_argument("--chains", type=int, default=12)
    ap.add_argument("--T", type=float, default=300.0)
    ap.add_argument("--ns", type=float, default=0.3)
    args = ap.parse_args()
    print(f"[polymer_md] {args.smiles}  mode={args.mode}  ({args.units}-mer x {args.chains})")
    if args.mode == "density":
        measure_density(args.smiles, n_units=args.units, n_chains=args.chains, T=args.T, ns=args.ns)
    elif args.mode == "tg":
        tg_scan(args.smiles, n_units=args.units, n_chains=args.chains, ns=args.ns)
    else:
        print("solubility mode: see solubility_parameter() — cohesive-energy run (todo v2)")

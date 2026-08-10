"""
test_reproducibility.py -- the sidebar's random seed must reproduce a run ACROSS restarts.

The GA seeds `random` and `numpy.random`, which makes a run reproducible within one process.
That is not enough: `build_seed_smiles()` collects base-polymer names into a set of STRINGS,
and Python randomises string hashing per process (PYTHONHASHSEED). Iterating that set
unsorted gave a different starting population in every interpreter, so the same seed produced
a different polymer after every app restart -- the sort in smart_ga.build_seed_smiles() is
what fixes it.

This test runs the seed-population build in child processes with DIFFERENT hash seeds and
requires an identical result. It loads no ML models, so it is fast.

    python tests/test_reproducibility.py
"""
import os, subprocess, sys, textwrap

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHILD = textwrap.dedent("""
    import hashlib, random, sys, warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, %r)
    import numpy as np, smart_ga as sga

    ACTIVE = ["Refractive", "Tg"]
    TARGETS = {"Refractive": 1.38, "Tg": 120.0}
    RANGES = {"Refractive": {"min": 1.2, "max": 1.8, "default": 1.5, "step": 0.01, "is_int": False},
              "Tg": {"min": -150.0, "max": 300.0, "default": 200.0, "step": 1.0, "is_int": False}}
    DATASET = ["[C][C][Branch1][C][C]"] * 50      # fixed stand-in for the seed population

    random.seed(int(sys.argv[1])); np.random.seed(int(sys.argv[1]))
    pop = sga.build_seed_population(ACTIVE, TARGETS, RANGES, DATASET)
    print(hashlib.sha1("\\n".join(pop).encode()).hexdigest())
""") % PROJ


def build(seed, hashseed):
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
    out = subprocess.run([sys.executable, "-c", CHILD, str(seed)],
                         cwd=PROJ, env=env, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"child failed (hashseed={hashseed}):\n{out.stderr[-1500:]}")
    return out.stdout.strip().splitlines()[-1]


def main():
    print("seed population must not depend on PYTHONHASHSEED")
    print("-" * 62)
    fails = 0

    for seed in (555, 42):
        digests = {h: build(seed, h) for h in (0, 1, 12345)}
        same = len(set(digests.values())) == 1
        fails += (not same)
        print(f"  seed {seed:<6} " + ("ok   " if same else "FAIL ")
              + "  ".join(f"hash{h}={d[:10]}" for h, d in digests.items()))
        if not same:
            print("     -> the same GA seed would give a different polymer after a restart")

    a, b = build(555, 0), build(42, 0)
    distinct = a != b
    fails += (not distinct)
    print(f"  different seeds still differ: {'ok' if distinct else 'FAIL - seeding collapsed'}")

    print("-" * 62)
    if fails:
        raise SystemExit(f"{fails} REPRODUCIBILITY CHECK(S) FAILED")
    print("ALL REPRODUCIBILITY CHECKS PASSED")


if __name__ == "__main__":
    main()

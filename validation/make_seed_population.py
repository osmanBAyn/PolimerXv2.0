"""
make_seed_population.py -- regenerate seed_population.json.gz.

The GA's starting population is a fixed list of polymer SMILES (and their SELFIES encodings)
taken from the training dataset. It never changes between runs, so downloading it from
HuggingFace on every container start was pure cost:

  * ~25 s before the first page could render,
  * all 18 splits of the repo downloaded to read one,
  * outbound network required from the container at runtime,
  * datasets + pyarrow + fsspec imported (~33 MB RSS) for a list of strings.

This script bakes that list into a 140 KB file instead. Run it only when the upstream dataset
changes; `datasets` is needed here but NOT by the deployed app.

    python validation/make_seed_population.py
"""
import os, sys, gzip, json, warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
os.chdir(PROJ)

REPO_ID = "OsBaran/Polimer-Ozellik-Tahmini"
OUT = os.path.join(PROJ, "seed_population.json.gz")


def main():
    from datasets import load_dataset
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    # Reuse the app's own encoder so the file matches what the live path produced exactly.
    from app import smiles_to_selfies_safe

    df = load_dataset(REPO_ID, split="Tg").to_pandas()
    col = "p_smiles" if "p_smiles" in df.columns else "smiles"
    smiles = df[col].tolist()
    selfies = [s for s in (smiles_to_selfies_safe(x) for x in smiles) if s]

    if len(selfies) < 5000 or len(smiles) < 5000:
        raise SystemExit(f"refusing to write a suspiciously small population "
                         f"({len(selfies)} selfies / {len(smiles)} smiles)")

    payload = {
        "source": f"{REPO_ID} (split='Tg')",
        "note": "Precomputed so the app needs no network at runtime.",
        "n_selfies": len(selfies), "n_smiles": len(smiles),
        "selfies": selfies, "smiles": smiles,
    }
    with gzip.open(OUT, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)

    with gzip.open(OUT, "rt", encoding="utf-8") as fh:
        back = json.load(fh)
    assert back["selfies"] == selfies and back["smiles"] == smiles, "round-trip mismatch"

    print(f"{len(selfies)} SELFIES / {len(smiles)} SMILES -> {OUT} "
          f"({os.path.getsize(OUT)/1024:.0f} KB), round-trip verified")


if __name__ == "__main__":
    main()

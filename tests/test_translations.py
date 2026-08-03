"""
test_translations.py — every user-visible string must exist in BOTH languages.

Two failure modes this catches:
  1. a lang_dict key present in one language but not the other;
  2. retro.py emitting a route type/mechanism with no Turkish entry (these are hardcoded
     English inside retro.py and are shown verbatim in the UI, so a missing entry silently
     leaves English text in the Turkish interface).

    python tests/test_translations.py
"""
import sys, os, re, ast

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
from lang_dict import LANGUAGES

FAIL = []

# ---------------------------------------------------------------- 1. key parity
tr, en = LANGUAGES["TR"], LANGUAGES["EN"]
only_tr = set(tr) - set(en)
only_en = set(en) - set(tr)
# retro route strings live in TR only on purpose: the key IS the English text and _()
# falls back to the key, so EN needs no entry.
retro_src = open(os.path.join(PROJ, "retro.py"), encoding="utf-8").read()


def _strings_in(node):
    """Every str constant reachable from a node (handles ternaries, concatenation, tuples)."""
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.append(n.value)
    return out


RETRO_LITERALS = set()
_tree = ast.parse(retro_src)
# dict literals: {"type": ..., "mechanism": ...}  (values may be ternaries)
for n in ast.walk(_tree):
    if isinstance(n, ast.Dict):
        for k, v in zip(n.keys, n.values):
            if isinstance(k, ast.Constant) and k.value in ("type", "mechanism"):
                RETRO_LITERALS.update(_strings_in(v))
# _META = {key: (Name, Mechanism, [roles])}  and  _classify_ab's return tuples
for n in ast.walk(_tree):
    if isinstance(n, ast.Tuple) and len(n.elts) >= 2:
        a, b = n.elts[0], n.elts[1]
        if (isinstance(a, ast.Constant) and isinstance(a.value, str)
                and isinstance(b, ast.Constant) and isinstance(b.value, str)
                and (a.value.startswith("Poly") or a.value.startswith("Condensation"))):
            RETRO_LITERALS.update((a.value, b.value))
SUFFIXES = {" (branched/crosslinked)",
            "; crosslinked/branched at the extra connection point(s)"}
allowed_tr_only = RETRO_LITERALS | SUFFIXES

unexpected = only_tr - allowed_tr_only
print(f"TR keys {len(tr)} | EN keys {len(en)}")
if unexpected:
    FAIL.append(f"keys only in TR (and not retro literals): {sorted(unexpected)}")
if only_en:
    FAIL.append(f"keys only in EN: {sorted(only_en)}")
print(f"  ok    key parity ({len(only_tr)} TR-only, all retro literals)"
      if not unexpected and not only_en else "  FAIL  key parity")

# ------------------------------------------------- 2. every retro literal has Turkish
print(f"\nretro.py emits {len(RETRO_LITERALS)} distinct type/mechanism strings")
missing = sorted(s for s in RETRO_LITERALS if s not in tr)
if missing:
    FAIL.append(f"retro strings with NO Turkish translation: {missing}")
    for m in missing:
        print(f"  FAIL  untranslated: {m}")
else:
    print("  ok    all retro route names/mechanisms have a Turkish entry")

for suf in SUFFIXES:
    if suf not in tr:
        FAIL.append(f"retro suffix not translated: {suf!r}")
print("  ok    branched suffixes translated" if all(s in tr for s in SUFFIXES)
      else "  FAIL  branched suffixes")

# --------------------------------------------- 3. untranslated (identical) TR entries
same = [k for k, v in tr.items()
        if k in en and v == en[k] and len(str(v)) > 12
        and k not in allowed_tr_only and not re.fullmatch(r"[\W\d_]+", str(v))]
if same:
    print(f"\n  NOTE  {len(same)} TR entries identical to EN (check if intentional):")
    for k in same[:10]:
        print(f"          {k}: {tr[k][:70]}")

# --------------------------------------------- 4. composed translation works end-to-end
sys.modules.setdefault("streamlit", None)
def fake_tr(text, lang="TR"):
    d = LANGUAGES[lang]
    hit = d.get(text, text)
    if hit != text:
        return hit
    for suf in SUFFIXES:
        if text.endswith(suf):
            return fake_tr(text[:-len(suf)], lang) + d.get(suf, suf)
    return text

print("\ncomposed-string translation:")
for probe in ["Polyester", "Polyester (branched/crosslinked)",
              "Vinyl / addition copolymer (branched/crosslinked)",
              "Step-growth condensation; crosslinked/branched at the extra connection point(s)"]:
    out = fake_tr(probe)
    ok = out != probe
    if not ok:
        FAIL.append(f"composed string not translated: {probe!r}")
    print(f"  {'ok  ' if ok else 'FAIL'}  {probe[:52]:54} -> {out[:56]}")

print("\n" + "=" * 60)
if FAIL:
    print(f"FAILED ({len(FAIL)}):")
    for f in FAIL:
        print("  -", f)
    sys.exit(1)
print("ALL TRANSLATION CHECKS PASSED")


def test_translations():        # pytest entry point
    assert not FAIL, FAIL

# Tests

Run these after touching `retro.py`, `smart_ga.py`, `app.py`, or `lang_dict.py`.
No pytest required — each file runs standalone and exits non-zero on failure.

```bash
python tests/test_retro.py         # retrosynthesis rules (fast, no models)
python tests/test_app_results.py   # app results path (loads the models, ~1 min)
```

## `test_retro.py`
Locks in retrosynthesis behaviour that real bugs previously violated:
- canonical polymers decompose to the right type **and** the right monomer
- **pendant groups never set the route type** — the backbone does (a pendant ester must not
  make PMMA a "polyester"; a pendant Si must not make a vinyl polymer a "silicone")
- every returned route is `verified` (the engine must never emit a wrong route)
- the round-trip `exact` / `approx` / `n/a` flag is correct
- junk input (`""`, `"*"`, `None`, unparseable) never raises

## `test_app_results.py`
Executes the whole app with a pre-seeded GA result, through Streamlit stubs, and asserts the
results path renders: the applicability-domain banner, per-property error bars, both download
buttons, and a complete text report (all sections present). This test caught a live crash
where `_` (the translation function) was shadowed by tuple unpacking — a whole class of bug in
this codebase, so keep it green.

**Never write `x, _ = something()` at module level in `app.py`** — `_` is the translation
function and shadowing it crashes the app at the next `_( )` call.

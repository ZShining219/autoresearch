# Paper workspace — realism-composed TSC benchmarking

This folder holds the manuscript draft, bibliography, extracted figures, and
research notes. It is **independent** of `result_analysis/` notebooks: those
notebooks are treated as read-only sources of tables/plots.

## Layout

| Path | Purpose |
|------|---------|
| `main.tex` | Full draft (intro → conclusion) |
| `references.bib` | BibTeX (LibSignal, MP, PressLight/CoLight, RESCO, Real Deal, robust RL, LLMLight, eval fragility, …) |
| `figures/` | PNGs extracted from notebook outputs + `MANIFEST.json` |
| `tables/` | Optional CSV/TeX tables for later polishing |
| `notes/CLAIMS_MAP.md` | Claim → evidence mapping |
| `notes/OPEN_QUESTIONS.md` | Clarifications needed from authors |
| `Makefile` | `make pdf` if `pdflatex`/`bibtex` available |

## Build

```bash
cd paper
make pdf    # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Do not

- Edit `result_analysis/*.ipynb` when refreshing figures; re-extract with the
  script below if notebooks change.
- Commit huge `data/output_data/` logs into the paper folder.

## Refresh figures from notebooks (read-only)

```bash
# from repo root
python3 - <<'PY'
# same extractor used to populate paper/figures/
# (re-run the extraction shell used in the paper drafting session)
PY
```

## Related repo docs

- `docs/REALISM_FULL.md`
- `docs/PARTIAL_OBSERVABILITY.md`
- `docs/CROSSING_PROXY.md`
- `docs/OD_HUB_DEMAND.md`
- `docs/TRAINING_GUIDE.md` (ghost physics)

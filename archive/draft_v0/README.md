# CrudeCortex-GWP3

Streamlit application for the GWP3 results, interpretation, validation, trading simulation and model-improvement work.

## Project layout

- `app.py` — Streamlit entry point
- `core/` — shared GWP2 engines plus GWP3 BDeu scoring and pipeline
- `pages/` — eight multipage app sections
- `data/fred/` — six FRED input CSVs
- `tests/test_gwp3.py` — GWP3 verification tests
- `documents/`, `notebooks/` — add the original report PDFs and notebooks here

## Setup

Use Python 3.11 or 3.12. From this folder:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Download data

The pipeline expects `WTISPLC.csv`, `POILBREUSDM.csv`, `INDPRO.csv`, `CPIENGSL.csv`, `DTWEXBGS.csv`, and `VIXCLS.csv` in `data/fred/`.

```bash
bash data/download_fred_data.sh
```

On Windows, run the script in Git Bash, or download each FRED CSV manually into `data/fred/`.

## Run

```bash
streamlit run app.py
```

## Tests

```bash
pytest -q tests/test_gwp3.py
```

The full tests require all six CSV files and the scientific Python dependencies.

## Important source completeness note

The uploaded GWP3 markdown includes the complete code for the BDeu engine, pipeline, app navigation, pages 1–4 and test file. It explicitly points to a previous GWP3 delivery for pages 5–8 but does not contain those source scripts. Those four files are marked as placeholders and must be replaced with the originals before treating the app as complete. The uploaded markdown also does not include the PDFs, notebook, or FRED datasets; add/download them before deployment.

Only Python syntax compilation is checked during packaging; runtime and data-dependent tests have not been executed.

Airport Wastewater & ICU Surveillance Modelling
==============================================

This repository contains code and data to explore early detection of emerging pathogens using:

- Airport wastewater surveillance, based on the `pgfgleam` Python package and associated notebooks.
- ICU-based genomic surveillance, based on the `NBPMscape.jl` Julia package and ICU workflow scripts.

The main end-to-end workflow combines international mobility data, epidemiological parameters, and a stochastic transmission model to estimate when outbreaks are first detected under different surveillance strategies.

The repository is organised so that you can:

- Reproduce the analyses from the paper (airport wastewater vs ICU).
- Plug in your own epidemiological parameters and your own mobility/network data.
- Run the ICU + wastewater detection simulations and export results.


## Repository layout

- `local_model/` – Julia package and examples for transmission and ICU surveillance modelling.
  - `src/` – core NBPMscape.jl code.
  - `NBPMscape_code/` – ICU + AWW examples.
  - `parameters.md` – detailed description of model parameters and literature sources.
- `global_model/` – Python package and notebooks for global mobility and arrival modelling.
  - `pgfgleam_code/` – Python library code and datasets.
  - `data_processing.py` – helpers for preparing our mobility data.
  - `simple_setup.ipynb` – user-facing notebook to plug in your own parameters and mobility data.
  - `figure*.ipynb`, `supp_*.ipynb` – analysis and plotting notebooks.
  - `global_model.ipynb` - setup and creation of CSVs for local_model usage


## Installation

You will need Julia (≥ 1.9 is recommended) and Python (≥ 3.9).

### Julia environment for NBPMscape

```julia
using Pkg
Pkg.activate("NBPMscape")
Pkg.instantiate()
```

Or from the shell:

```bash
cd /Users/reddy/AWW_and_ICU/NBPMscape
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This sets up all Julia dependencies required for the ICU + wastewater simulations.

### Python environment for pgfgleam

From the shell:

```bash
cd /Users/reddy/AWW_and_ICU/pgfgleam
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

The editable install (`-e .`) makes the `pgfgleam` package and helper scripts (including `data_processing.py`) importable inside notebooks.

If you prefer, you can reuse the existing `wastewater_venv/` environment instead of creating `.venv`.


## Quick start: run the ICU + wastewater workflow

The end-to-end ICU + wastewater workflow proceeds in two broad stages:

1. Prepare daily import trajectories for each country and parameter combination (Python, `pgfgleam`).
2. Run ICU + wastewater detection simulations on those import trajectories (Julia, `NBPMscape`).

### 1. Prepare daily imports with `pgfgleam`

Open the notebook:

- `pgfgleam/simple_setup.ipynb`

In this notebook you can:

- Point to your own mobility / flight matrix, population data, and country-code mappings (via `DataSetup` in `data_processing.py`).
- Specify epidemiological parameters (e.g. $R_0$, generation time, infectious period).
- Generate a CSV of daily introduction trajectories in the format expected by the ICU + wastewater workflow (by default saved as:
  - `pgfgleam/daily_imports_sensitivity.csv`).

You can also start from the existing `daily_imports_sensitivity.csv` in `pgfgleam/` as a template and adjust it.

### 2. Run ICU + wastewater simulations with Julia

Once `daily_imports_sensitivity.csv` is prepared, run the ICU + wastewater simulations using:

- `NBPMscape/AWW_ICU_examples/full_ICU_WW.jl`

From the shell:

```bash
cd /Users/reddy/AWW_and_ICU/NBPMscape
julia --project=. AWW_ICU_examples/full_ICU_WW.jl
```

In its current configuration, this script:

- Reads the merged imports CSV (by default `pgfgleam/daily_imports_sensitivity.csv`).
- Runs the ICU-based detection and airport wastewater detection simulations across a grid of parameters.
- Periodically saves results to a CSV in `pgfgleam/datasets/` (e.g. `full_result.csv`), which you can analyse in R, Python, or Julia.

The script uses a batched, parallelised simulation strategy (see the top of `full_ICU_WW.jl` for details and resource requirements, particularly `addprocs(…)`).


## Using `simple_setup.ipynb` with your own data

The `simple_setup.ipynb` notebook is meant to be the main entry point for external users. A typical workflow is:

1. Set paths to your data
   - Flight / mobility matrix (e.g. monthly flows between countries or regions).
   - Population data (e.g. UN WPP population by country).
   - Country code / name mapping file.
2. Run the data-preparation cell
   - Uses the `DataSetup` class in `data_processing.py` to normalise flows by population and harmonise country names.
3. Choose epidemiological parameters
   - $R_0$ values, generation-time distributions, latent and infectious periods, ICU sampling fraction, etc.
   - These are documented with literature references in `NBPMscape/parameters.md`.
4. Export the daily imports CSV
   - Notebook saves a CSV compatible with `full_ICU_WW.jl` (one row per country / parameter combination / time).
5. Run the Julia script
   - Either from the command line (recommended) or via a `subprocess` call shown at the end of the notebook.

The notebook is deliberately written to be transparent and hackable, so you can adapt it to other surveillance or outbreak scenarios.


## Licences and attribution

- Top-level repository: MIT License (see `LICENSE` in the repository root).
- `NBPMscape/`: MIT-licensed Julia package by Kieran Drake; see `NBPMscape/LICENSE` and the upstream project documentation at  
  `https://github.com/emvolz/NBPMscape.jl`.
- `pgfgleam/`: MIT-licensed Python package by Guillaume St-Onge; see `pgfgleam/LICENSE`.

If you use this repository in academic work, please:

- Cite the NBPMscape and pgfgleam projects as appropriate.
- Cite the epidemiological and methodological references listed in `NBPMscape/parameters.md`.


## Reproducibility notes

- Some scripts (e.g. `full_ICU_WW.jl`) make heavy use of parallel processing. Adjust `addprocs(...)` to match your local or cluster resources.
- Large intermediate CSV files (mobility matrices, country-level arrivals, full simulation outputs) can be sizeable; for GitHub you may want to:
  - Exclude them via `.gitignore`, or
  - Store them separately (e.g. Zenodo, OSF) and document download links.

The code is structured so that all key analyses can be regenerated from source given:

- A suitable mobility dataset;
- The parameter choices documented in `parameters.md`; and
- Adequate compute resources for the stochastic simulations.


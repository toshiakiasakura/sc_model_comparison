# Survey design matters for capturing heterogeneity in social contacts: from exponential to scale-free

This repository contains code and data for analyzing social contact survey data and comparing different statistical models to capture heterogeneity in contact patterns.

## System Requirements

### Software Dependencies
- **Julia**: Version 1.11.1 or higher (tested on Julia 1.11.1)
- **Jupyter**: For running analysis notebooks (installed via IJulia.jl)

### Julia Package Dependencies
All Julia package dependencies are specified in `Project.toml` and `Manifest.toml`, including:
- Turing.jl, Pathfinder.jl — Bayesian inference (NUTS/MCMC with Pathfinder initialization)
- Distributions.jl — probability distributions
- JLD2.jl — serialization of fitted model chains
- DataFrames.jl, CSV.jl, XLSX.jl — data manipulation
- Plots.jl, StatsPlots.jl — plotting
- RCall.jl — natural splines (R `splines::ns`) for the sample-size meta-regression

All versions are pinned in `Manifest.toml`. The Manifest is for Julia 1.11.1;
newer Julia versions (e.g. 1.12) cannot load it, so use Julia 1.11.1 or the dev container.

R is also required (for RCall.jl). The dev container installs it.

### Hardware Requirements
- **RAM**: 16GB minimum. The CoMix2 bootstrap fits in `4j` peak at about 24GB at the
  largest sample size (see the comments on `fit_bootstrap_data_CoMix2` in `src/comix2_utils.jl`).
- **Storage**: At least 2GB free disk space for data and intermediate results
  (plus about 1.2GB if the bootstrap results are regenerated)
- **CPU**: Multi-core recommended (12 threads used by default for parallel MCMC fitting)
- No GPU or other non-standard hardware required

### Tested Platforms
- Windows 11 with Julia 1.11.1
- macOS (Apple Silicon) with the dev container (Julia 1.11.1)

## Installation Guide

### Option 1: Using VS Code with Dev Container (Recommended)
1. Install Docker Desktop
2. Install VS Code with the Dev Containers extension
3. Clone this repository:
   ```bash
   git clone <repository-url>
   cd prj_sc_model_comparison
   ```
4. Open the folder in VS Code
5. Press `Ctrl + Shift + P` (Windows/Linux) or `Cmd + Shift + P` (macOS)
6. Select `Dev Container: Reopen in Container`
7. Wait for the container to build and start

**Typical install time**: 10–15 minutes (depending on internet speed and container build)

### Option 2: Manual Setup
1. Install Julia 1.11.1
2. Clone the repository and install dependencies:
   ```bash
   git clone <repository-url>
   cd prj_sc_model_comparison
   ```
   ```julia
   using Pkg; Pkg.instantiate()
   ```
3. Start Jupyter:
   ```bash
   julia -e "using IJulia; notebook()"
   ```

## Instructions for Use

### Running the Full Analysis

The analysis pipeline consists of Jupyter notebooks in `src/` that must be run sequentially:

1. **Data Setup** (`src/1j_data_setup.ipynb`):
   - Loads raw survey data from `dt_surveys/`, `dt_Leon_Danon_2013/` and the metadata in `dt_surveys_master/`
   - Builds the adult (age 18+) degree distributions for "all", "home" and "non-home" contacts
   - Writes them to `dt_surveys_master/master_dds.csv`

2. **Fit Survey Models** (`src/2j_fit_surveys.ipynb`):
   - Fits three competing zero-inflated models (ZInf-NB, ZInf-PLN, ZInf-PLomax) per survey and contact setting
   - Uses Bayesian inference with Turing.jl (NUTS sampler with Pathfinder initialization)
   - Saves fitted chains to `dt_intermediate/<key>_chns.jld2`
   - Estimates the extreme value index (EVI) at thresholds q = 0.95, 0.97 and 0.98 (`EVI_QS`)
     and writes `dt_intermediate/EVI_summary.csv`
   - **Runtime: 1–2 days** for a fit from scratch
   - `fit_surveys(df_dds)` skips surveys that already have a chain file. After changing
     `master_dds.csv`, run `fit_surveys(df_dds; refit_stale = true)`. This refits only the
     surveys whose degree distributions changed (`is_fit_stale`).
   - Check convergence with `is_chains_converged` (ESS > 200 and R-hat < 1.1).
     `refit_unconverged!(key)` refits failing chains with 4,000 and then 8,000 draws.

3. **Meta-Regression Analysis** (`src/3j_meta_reg.ipynb`):
   - Summarises each fit (means, WAIC, convergence) in `dt_intermediate/fit_summary.csv`.
     After any refit, rebuild it with `read_master_with_fit_summary(; update_data = true)`.
   - Performs meta-analysis across surveys using WAIC weights
   - Compares different model specifications via fractional multinomial logit
   - The combined "Danon 2013" and "Leung 2017" sets are excluded (`filter_surveys`);
     their paper/online subsets are analysed instead.

4. **Additional CoMix Analysis** (`src/4j_comix2_additional.ipynb`):
   - CoMix-specific bootstrap resampling and uncertainty quantification
   - Convolution of home and non-home distributions, truncated at K = 20,000
   - Outputs to `dt_intermediate_bootstrap/`
   - Simulations and bootstrap data are seeded (`seed` keyword, default 1234)
   - The bootstrap fits are commented out in the notebook. They take several hours
     (about 8 hours at the largest sample size alone), and their saved results are reused.

5. **Supplementary Figures** (`src/5j_supplementary.ipynb`):
   - Generates supplementary figures for the paper

Apart from the bootstrap fits, notebooks 1, 3, 4, and 5 complete within 30 minutes each.

### Input Data Formats

#### Survey Data (in `dt_surveys/` and `dt_surveys_master/`)
The software expects standardized survey data files with the following formats:

- **Contact data** (`*_contact_common.csv`):
  - Columns: participant_id, contact_id, age_participant, age_contact, duration, location, etc.
  - CSV format with UTF-8 encoding

- **Participant data** (`*_participant_common.csv`):
  - Columns: participant_id, age, gender, household_size, day_of_week, etc.
  - CSV format with UTF-8 encoding

- **Household data** (`*_hh_common.csv`):
  - Columns: household_id, size, composition, etc.
  - CSV format with UTF-8 encoding

- **Survey metadata**: `dt_surveys_master/survey_mastersheet_info.xlsx`
  (file names, survey mode, group contacts, contact caps; read by `read_survey_master_data`)

#### Data sources and handling notes
- **CoMix UK** data are from Zenodo ([10.5281/zenodo.6542524](https://doi.org/10.5281/zenodo.6542524)).
  The analysis uses adults surveyed from July 2021 to February 2022.
  The local `comix_uk_*.csv` files are identical to the Zenodo files; only the file names are in lower case.
- **Do not open and save the raw CSV files in Excel.** Excel keeps only the first
  1,048,576 rows. `comix_uk_contact_common.csv` has 1,307,501 rows, and an earlier
  copy of it was silently truncated this way.
- **Missing home/non-home flag:** a participant-day with any contact whose `cnt_home` is missing
  is kept in "all" but excluded from "home" and "non-home". For these surveys, the sample size
  for "all" can therefore exceed the sample size for home/non-home.
- **Contacts reported outside the diary** are added to "all" and "non-home"
  (`add_extra_counts` in `src/data_setup.jl`):
  - Beraud 2015: supplementary professional contacts, weekdays only
  - Zhang 2019: group contacts (`group_n`)
- **Danon 2013:** home contacts are counted in people, using the group size `C_Number`, so they
  match `P_total_contacts`. Non-home is total minus home.

### Output Data Formats

#### Intermediate Results (`dt_intermediate/`)
- **JLD2 files** (`*_chns.jld2`): Serialized Julia objects containing fitted model chains, posterior distributions, and degree distributions

#### Bootstrap Results (`dt_intermediate_bootstrap/`)
These files are not uploaded because of their size.
- **JLD2 files** (`comix2_*samples_*repeat.jld2`): Bootstrap resampling results
- **CSV files** (`comix2_waic_weights_boot.csv`): Model weights and comparison statistics

#### Figures (`fig/`)
- PNG files: Publication-ready figures

## Project Structure

```
src/                          <- Analysis notebooks and Julia source files
  main_utils.jl               <- Top-level include; imports packages and all utilities
  distributions/              <- Custom probability distributions
    poisson_mixture.jl        <- NegBin, PoissonLogNormal, PoissonLomax, Lomax
    zeroinf.jl                <- Zero-inflated distribution wrappers
    zerotrunc.jl              <- Zero-truncated variants
    kernel.jl, helper.jl, plot.jl
  utils.jl                    <- General helpers
  degree_dist.jl              <- DegreeDist struct + PDF/CCDF plotting
  turing_models.jl            <- Turing.jl @model definitions
  turing_utils.jl             <- Parameter extraction from MCMC chains, convergence check
  fit_utils.jl                <- Fitting/refitting, WAIC, EVI estimation, meta-regression plotting
  plot_utils.jl               <- Additional plotting utilities
  data_setup.jl               <- Raw data loading and preprocessing (included separately)
  comix2_utils.jl             <- CoMix2 simulation, bootstrap, convolution and age-block analyses
  suppl_utils.jl              <- Fit summary and supplementary/validation figures
dt_surveys/                   <- Raw survey CSV files
dt_Leon_Danon_2013/           <- Danon 2013 contact and person data
dt_surveys_master/            <- Survey metadata and the combined degree distributions (master_dds.csv)
dt_intermediate/              <- Fitted model chains (JLD2), fit_summary.csv, EVI_summary.csv
dt_intermediate_bootstrap/    <- Bootstrap results
fig/                          <- Output figures
```

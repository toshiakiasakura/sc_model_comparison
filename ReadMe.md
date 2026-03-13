# Survey design matters for capturing heterogeneity in social contacts: from exponential to scale-free

This repository contains code and data for analyzing social contact survey data and comparing different statistical models to capture heterogeneity in contact patterns.

## System Requirements

### Software Dependencies
- **Julia**: Version 1.11.1 or higher (tested on Julia 1.11.1)
- **Jupyter**: For running analysis notebooks (installed via IJulia.jl)

### Julia Package Dependencies
All Julia package dependencies are specified in `Project.toml` and `Manifest.toml`, including:
- Turing.jl — Bayesian inference (NUTS/MCMC)
- Distributions.jl — probability distributions
- JLD2.jl — serialization of fitted model chains
- DataFrames.jl, CSV.jl — data manipulation
- CairoMakie.jl — plotting

All versions are pinned in `Manifest.toml`.

### Hardware Requirements
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: At least 2GB free disk space for data and intermediate results
- **CPU**: Multi-core recommended (12 threads used by default for parallel MCMC fitting)
- No GPU or other non-standard hardware required

### Tested Platforms
- Windows 11 with Julia 1.11.1

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
   - Loads raw survey data from `dt_surveys/` and `dt_surveys_master/`
   - Processes and standardizes contact survey data
   - Outputs processed data to `dt_intermediate/`

2. **Fit Survey Models** (`src/2j_fit_surveys.ipynb`):
   - Fits three competing zero-inflated models (ZInf-NB, ZInf-PLN, ZInf-PLomax) per survey and contact setting
   - Uses Bayesian inference with Turing.jl (NUTS sampler with Pathfinder initialization)
   - Saves fitted chains to `dt_intermediate/`
   - **Runtime: 1–2 days**

3. **Meta-Regression Analysis** (`src/3j_meta_reg.ipynb`):
   - Performs meta-analysis across surveys using WAIC weights
   - Compares different model specifications via fractional multinomial logit

4. **Additional CoMix Analysis** (`src/4j_comix2_additional.ipynb`):
   - CoMix-specific bootstrap resampling and uncertainty quantification
   - Outputs to `dt_intermediate_bootstrap/`

5. **Supplementary Figures** (`src/5j_supplementary.ipynb`):
   - Generates supplementary figures for the paper

Notebooks 1, 3, 4, and 5 complete within 30 minutes each.

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

- **Survey metadata**: `dt_surveys_master/master_line_data_v1.csv`

### Output Data Formats

#### Intermediate Results (`dt_intermediate/`)
- **JLD2 files** (`*_chns.jld2`): Serialized Julia objects containing fitted model chains, posterior distributions, and degree distributions

#### Bootstrap Results (`dt_intermediate_bootstrap/`)
This files are not uploaded due to a file size.
- **JLD2 files** (`comix2_*samples_*repeat.jld2`): Bootstrap resampling results
- **CSV files** (`comix2_waic_weights.csv`): Model weights and comparison statistics

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
  utils.jl                    <- General helpers
  degree_dist.jl              <- DegreeDist struct + PDF/CCDF plotting
  turing_models.jl            <- Turing.jl @model definitions
  turing_utils.jl             <- Parameter extraction from MCMC chains
  fit_utils.jl                <- WAIC, EVI estimation, meta-regression plotting
  plot_utils.jl               <- Additional plotting utilities
  data_setup.jl               <- Raw data loading and preprocessing
dt_surveys/                   <- Raw survey CSV files
dt_surveys_master/            <- Survey metadata
dt_intermediate/              <- Fitted model chains (JLD2)
dt_intermediate_bootstrap/    <- Bootstrap results
fig/                          <- Output figures
```

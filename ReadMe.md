# Survey design matters for capturing heterogeneity in social contacts: from exponential to scale-free

This repository contains code and data for analyzing social contact survey data and comparing different statistical models to capture heterogeneity in contact patterns.

This repository is still under development.

## System Requirements
### Software Dependencies
This software requires the following:
- **Julia**: Version 1.11.1 or higher
  - Tested on: Julia 1.11.1

### Julia Package Dependencies
All Julia package dependencies are specified in `Project.toml` and `Manifest.toml`, including:
- Distributions (v0.25 or higher)
- Turing (v0.30 or higher)
- JLD2 (v0.4 or higher)
- And other dependencies listed in `Project.toml`

### Hardware Requirements
- **Standard Computer**: 8GB RAM minimum, 16GB recommended
- **Storage**: At least 2GB free disk space for data and intermediate results
- **No special hardware required**: No GPU or other non-standard hardware needed

### Tested Platforms
This software has been tested on:
- Windows 11 with Julia 1.11.1

## Installation Guide
### Installation Steps

#### Option 1: Using VS Code with Dev Container (Recommended)
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

**Typical install time**: 10-15 minutes (depending on internet speed and container build)

## Demo
## Instructions for Use
### Running the Full Analysis
The analysis pipeline consists of several sequential steps:
1. **Data Setup** (`src/1j_data_setup.ipynb`):
   - Loads raw survey data from `dt_surveys/` and `dt_surveys_master/`
   - Processes and standardizes contact survey data
   - Outputs processed data to `dt_intermediate/`

2. **Fit Survey Models** (`src/2j_fit_surveys.ipynb`):
   - Fits statistical models to individual survey datasets
   - Uses Bayesian inference with Turing.jl
   - Saves fitted models and chains to `dt_intermediate/`

3. **Meta-Regression Analysis** (`src/3j_meta_reg.ipynb`):
   - Performs meta-analysis across surveys
   - Compares different model specifications
   - Generates model comparison metrics

4. **Additional CoMix Analysis** (`src/4j_comix2_additional.ipynb`):
   - Specific analysis for CoMix survey data
   - Bootstrap resampling and uncertainty quantification

The code for `src/2j_fit_surveys.ipynb` will take 1 day to 2 days to
complete while other codes would be finished within 30 minutes.

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

### Output Data Formats
#### Intermediate Results (`dt_intermediate/`)
- **JLD2 files** (`*_chns.jld2`): Serialized Julia objects containing:
  - Fitted model chains from Bayesian inference
  - Posterior distributions
  - Model comparison metrics (WAIC, LOO)

#### Bootstrap Results (`dt_intermediate_bootstrap/`)
- **JLD2 files** (`comix2_*samples_*repeat.jld2`): Bootstrap resampling results
- **CSV files** (`comix2_waic_weights.csv`): Model weights and comparison statistics

#### Figures (`fig/`)
- PNG/PDF files: Publication-ready figures
- Organized in subdirectories (e.g., `1x/` for standard resolution)

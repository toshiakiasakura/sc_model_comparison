# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Julia/Jupyter research project analyzing social contact survey data to compare statistical models for heterogeneity in contact patterns. The paper title is: *"Survey design matters for capturing heterogeneity in social contacts: from exponential to scale-free"*.

## Running the Analysis

The analysis pipeline must be run sequentially via Jupyter notebooks (`src/`):

1. `1j_data_setup.ipynb` — loads raw survey data → outputs to `dt_intermediate/`
2. `2j_fit_surveys.ipynb` — Bayesian model fitting (NUTS/MCMC), **takes 1–2 days**
3. `3j_meta_reg.ipynb` — meta-regression across surveys
4. `4j_comix2_additional.ipynb` — CoMix-specific bootstrap analysis
5. `5j_supplementary.ipynb` — supplementary figures
6. `6j_data_preparation.ipynb` — additional data prep

To start Jupyter with Julia kernel:
```bash
julia -e "using IJulia; notebook()"
```

To instantiate the Julia environment (first time):
```julia
using Pkg; Pkg.instantiate()
```

## Architecture

### Source file structure (`src/`)

All notebooks share a common set of `.jl` utility files loaded via `main_utils.jl`:

```
main_utils.jl        ← top-level include, imports all packages and includes all utils
  distributions/main.jl  ← custom probability distributions
    poisson_mixture.jl   ← NegBin, PoissonLogNormal, PoissonLomax, Lomax
    zeroinf.jl           ← ZeroInfDist, ZeroInfConvolutedDist (zero-inflated wrappers)
    zerotrunc.jl         ← zero-truncated variants
    kernel.jl, helper.jl, plot.jl
  utils.jl           ← general helpers (read_survey_master_data, model_abbr dict, etc.)
  degree_dist.jl     ← DegreeDist struct + PDF/CCDF plotting functions
  turing_utils.jl    ← parameter extraction from MCMC chains, distribution constructors
  turing_models.jl   ← all Turing.jl @model definitions
  fit_utils.jl       ← post-processing: WAIC, EVI estimation, meta-regression plotting
  plot_utils.jl      ← additional plotting utilities
  data_setup.jl      ← raw data loading and preprocessing helpers
```

Notebooks include `main_utils.jl` at the top, which pulls in everything else.

### Core data structure

`DegreeDist` (`degree_dist.jl`) is the central data type representing a contact degree distribution as paired vectors `(x::Vector{Int64}, y::Vector{Int64})` — degrees and counts respectively. Most fitting and plotting functions accept `DegreeDist`.

### Statistical models (`turing_models.jl`)

Three competing zero-inflated models are fit per survey × contact setting (home/non-home):
- `ZeroInfNegativeBinomial` (ZInf-NB) — parameters: `log_m_ga`, `log_k_ga`, `π0`
- `ZeroInfPoissonLogNormal` (ZInf-PLN) — parameters: `μ_obs_ln`, `log_σ_ln`, `π0`
- `ZeroInfPoissonLomax` (ZInf-PLomax) — parameters: `log_α_lo`, `log_β_lo`, `π0`

Model fitting uses NUTS sampler with Pathfinder initialization (via `get_median_parms_from_model`) and multi-threaded fitting across model × setting combinations (`Threads.@threads`).

### Data flow

- Raw survey CSVs: `dt_surveys/` and `dt_surveys_master/`
- Survey metadata: `dt_surveys_master/survey_mastersheet_info.xlsx`
- Fitted chains (JLD2): `dt_intermediate/<key>_chns.jld2` — contains `Dict` with `"dds"`, `"chns_home"`, `"chns_non-home"` keys
- Bootstrap results: `dt_intermediate_bootstrap/`
- Figures: `fig/`

### Model comparison

WAIC is computed in `fit_utils.jl:calc_waic`. WAIC weights are used for meta-regression via a fractional multinomial logit model (`model_fmnl` in `turing_models.jl`). The `flag_minimum_IC` function adds `weight_waic` columns to results DataFrames.

### Convergence check

Use `is_chains_converged(chn)` — checks ESS > 200 and R-hat < 1.1 for all parameters. Use `explore_chns(chn)` for interactive diagnostics.

## Key Conventions

- Parameters are stored in log-space during sampling; `exponential_convert` / `PoissonLogNormal_convert` transform them back
- Contact settings: `"home"`, `"non-home"`, `"all"`; survey keys are string identifiers like `"comix2_2020_1"`
- `@memoize` is used on expensive distribution evaluations (`logpdf`, `ccdf`) to avoid recomputation
- All file paths in notebooks and scripts are relative to `src/` (e.g., `"../dt_intermediate/"`)
- Julia version: 1.11.1; all dependencies pinned in `Manifest.toml`

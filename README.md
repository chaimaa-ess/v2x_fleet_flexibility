# EV Fleet V2X Flexibility: Estimation, Forecasting and Decision Support

Code accompanying the manuscript *An Application-Independent Framework for EV Fleet V2X Flexibility Estimation and Forecasting* 
(submitted to Applied Energy).

## Overview

This repository implements a unified framework for estimating and forecasting aggregate electric vehicle (EV) fleet flexibility. Individual EV operating constraints are aggregated into a compact, time-coupled polytope. Its power and cumulative-energy bounds can be forecast and reused as linear constraints in different market and grid optimisation problems without application-specific flexibility models.

## Repository navigation

| Folder or file | Contents |
| --- | --- |
| `v2x_flex_estimation/` | Polytope estimation, aggregation comparison, and flexibility-contracting case study |
| `v2x_flex_forecasting/` | Forecasting experiments and evaluation |
| `v2x_flex_forecasting/src/` | Shared forecasting and data-processing code |
| `v2x_flex_forecasting/run_window_search.py` | Historical-window selection |
| `v2x_flex_forecasting/run_screening.py` | Forecasting model screening |
| `v2x_flex_forecasting/run_deep_search.py` | Focused model tuning |
| `v2x_flex_forecasting/run_final_evaluation.py` | Final model evaluation |
| `v2x_flex_forecasting/doe_case_study/` | Day-ahead scheduling with dynamic operating envelopes |
| `v2x_flex_forecasting/results/` | Saved evaluation outputs and figures |
| `v2x_flex_forecasting/results/final_forecasts.zip` | Final forecast outputs |

## Suggested navigation

1. Start with the estimation notebook in `v2x_flex_estimation/` to inspect EV battery constraints, fleet aggregation, and approximation evaluation.
2. Explore `v2x_flex_forecasting/src/` and the `run_*.py` scripts for forecasting methodology.
3. Consult `v2x_flex_forecasting/results/` for evaluation results.
4. Review the flexibility-contracting in the v2x_flex_estimation notebook and `doe_case_study/` for the two decision-support demonstrations.


## Software environments

The estimation and forecasting components were developed using separate Python environments.

### EV fleet flexibility estimation

The estimation notebook `v2x_flex_estimation/v2x_flex_estimation.ipynb` uses the OPLEM Python environment, hosted by PSAL Laboratory.

Installation instructions and dependencies are available in the official OPLEM repository: https://github.com/PSALOxford/OPLEM
- Instead of mosek, the notebook uses **Gurobi 13.0 Acaademic License** as an optimisation solver. 

### EV fleet flexibility forecasting

The forecasting experiments were developed in a separate Python environment: `requirements_forecasting.txt` file .

## Data availability

The datasets and data files used by the forecasting framework are organised under: `v2x_flex_forecasting/Data/`

The original data sources and their corresponding references are provided in the accompanying manuscript.

## Citation

If you use this repository, please cite the associated manuscript

## Contact

Please use the repository's GitHub Issues page for questions and reproducibility issues.

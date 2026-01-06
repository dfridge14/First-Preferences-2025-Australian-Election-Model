# First Preferences 2025 Australian Election Model

This repository contains the code used to simulate First Preference vote distributions and seat outcomes for the 2025 Australian federal election.

The model combines pre-generated polling data and outputs of the fundamentals model, together with historical election results and preference flows to generate probabilistic seat outcomes.

## Requirements

- Python 3.10 or newer

Install dependencies: `pip install -r requirements.txt`

## Data

This repository relies on a set of prerequisite CSV input files.

Place all required input data in: `data/raw/`

Intermediate and generated files (e.g. simulation outputs, summaries, and plots) are written to: `data/generated/`

These files are intentionally excluded from version control.

## Running the model

To run the full model end-to-end: `python Simulate_Model_from_Fundamentals_Polling.py`

The script executes the complete pipeline, including:

- Estimation of polling and election swing covariance structures
- Simulation of First Preference vote distributions
- Distribution of preferences to determine final seat outcomes
- Generation of summary tables and visualisations

## Outputs

Model outputs include:

- Simulated First Preference vote distributions
- Per-electorate and per-simulation seat winners
- Aggregate seat distributions by party

Outputs are written to `data/generated/` and can be used directly for downstream analysis or website visualisation.

Code in the `internal/` directory is being progressively migrated into the main project structure as more model features are exposed for general use.

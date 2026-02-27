# Project Pipeline

## 1. Data Preparation
Data cleaning: https://github.com/lanhung/material-conductivity-data-clean  
First run `data_loader.py` to fully sync the data from MySQL. (A snapshot has already been placed in the `data` folder as `zirconia_snapshot.duckdb`.)

## 2. Core Execution Steps

### Step 0: Initialization
- **Script**: `00_init_dir.py`
- **Purpose**: Initialize the project directory structure and create required output folders such as `results/`, `models/`, `logs/`, etc.

### Step 1: Foundation Training
- **Script**: `01_train_physics_model.py`
- **Purpose**: Train the core **Physics-Informed Machine Learning (PIML)** model.
- **Output**: Saves `best_piml_model.pth` under `results/checkpoint/piml/`.
- **Importance**: The cornerstone of the entire project—every downstream analysis and discovery module depends on this model.

### Step 2: Deep Analysis & Full-Data Model
- **Scripts**: `02_interpret_mechanisms.py`, `inspect_text_svd.py`
- **Purpose**: Perform feature-importance analysis, t-SNE visualization, and fine-tune the model using the full dataset.
- **Output**: Explanatory plots and `piml_full_model.pth`.

### Step 3: Benchmarking & Evaluation
This step includes two scripts to demonstrate the model’s advantages:
1. **`03a_train_baseline_model.py`**:
    - **Purpose**: Train a standard DNN (without physics constraints) as a baseline.
    - **Output**: `best_baseline_dnn.pth`.
2. **`03b_evaluate_benchmarks.py`**:
    - **Purpose**: Compare PIML against DNN, Random Forest (RF), and XGBoost, and run virtual screening tests.

### Step 4: Inverse Design & Theory Verification
- **Script**: `04_discover_materials.py`
- **Dependency**: Requires a model produced by Step 1 or Step 2.
- **Purpose**: Use a **Genetic Algorithm (GA)** to perform inverse design for co-doping recipes, search for new high-conductivity materials, and verify the “lattice strain theory.”
- **Output**: The best recipe file `results/ai_discovery_best_recipe.csv`.

### Step 5: Lab Simulation (Active Learning Demo)
- **Script**: `05_simulate_lab_experiments.py`
- **Note**: A standalone demo module.
- **Purpose**: Simulate an “Active Learning” loop with “Uncertainty Quantification (UQ)” to demonstrate how AI can reduce experimental iterations.

### Step 6: ML-Based Stability Pre-Screening
- **Script**: `06_verify_stability.py`
- **Purpose**: Quickly predict phase stability with an ML model and filter out unstable candidate recipes.

### Step 7: Molecular Dynamics Validation (MD with CHGNet)
- **Script**: `07_computational_validation.py`
- **Tech**: Integrated CHGNet (a general-purpose graph neural network interatomic potential).
- **Purpose**: Build a supercell and run high-temperature MD simulations (e.g., 1500K). Compute oxygen-ion mean squared displacement (MSD) to estimate diffusion coefficients and ionic conductivity, and compare against PIML predictions.
- **Output**: A PIML vs MD validation figure `results/images/paper_computational_validation.png`.

### Step 8: First-Principles Validation (DFT with Quantum Espresso)
- **Script**: `08_verify_stability_dft.py`
- **Tech**: Density Functional Theory (DFT).
- **Purpose**: Auto-generate Quantum Espresso (`pw.x`) input files and compute formation energy for candidate materials to confirm thermodynamic stability at the quantum-mechanical level.
- **Output**: `.pw.in` input files and a stability-analysis bar chart.

import os.path

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler  # [New] For baseline temperature scaling
import warnings

# Import config paths
from config import path_config
from etl.material_data_processor import MaterialDataProcessor
from features.preprocessor import build_feature_pipeline
from models.piml_net import PhysicsInformedNet
# [New] Import baseline model definition
from models.baseline_net import StandardDNN

# Suppress warnings
warnings.filterwarnings('ignore')

# ================= 1. Config & parameters =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ================= 2. Data preparation =================
def get_data_and_pipeline():
    """
    Load data and build the feature engineering pipeline.
    """
    print(">>> [Setup] Loading data via MaterialDataProcessor...")
    processor = MaterialDataProcessor()
    df = processor.load_and_preprocess_data_for_training_piml()

    # Split train/test set
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)

    # Build and fit pipeline
    print(">>> [Setup] Fitting feature pipeline...")
    pipeline = build_feature_pipeline()
    X_train = pipeline.fit_transform(train_df)
    X_test = pipeline.transform(test_df)

    # Extract target variable (log_conductivity)
    target_col = 'log_conductivity'
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    return train_df, test_df, X_train, X_test, y_train, y_test, pipeline

# ================= 3. Module 1: benchmark comparison =================
def run_benchmark(X_train, y_train, X_test, y_test):
    print("\n>>> [Module 1] Running Benchmarks (RF & XGBoost)...")
    results = {}

    # --- Model 1: Random Forest ---
    rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results['Random Forest'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'R2': r2_score(y_test, y_pred_rf)
    }

    # --- Model 2: XGBoost ---
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, n_jobs=-1)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    results['XGBoost'] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
        'R2': r2_score(y_test, y_pred_xgb)
    }

    return results

# ================= 4. Module 2: PIML physical analysis =================
def analyze_physics(model, X_test, test_df):
    """
    Check whether the physical parameters learned by the PIML model (Ea, logA) are consistent with electrochemical expectations.
    """
    print("\n>>> [Module 2] Analyzing Physics Consistency...")
    model.eval()

    # Prepare inputs
    X_tensor = torch.FloatTensor(X_test).to(DEVICE)
    # Note: take the temperature column from the DataFrame and convert to a tensor.
    T_tensor = torch.FloatTensor(test_df['temperature_kelvin'].values).view(-1, 1).to(DEVICE)

    with torch.no_grad():
        preds, Ea_pred, logA_pred = model(X_tensor, T_tensor)

    # Attach predicted physical parameters back to the DataFrame for plotting
    analysis_df = test_df.copy()
    analysis_df['pred_Ea'] = Ea_pred.cpu().numpy().flatten()
    analysis_df['pred_logA'] = logA_pred.cpu().numpy().flatten()
    analysis_df['pred_log_sigma'] = preds.cpu().numpy().flatten()

    # --- Plot 1: activation energy vs ionic radius (lattice distortion effect) ---
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    # Show only top dopants to avoid an oversized legend
    top_dopants = analysis_df['primary_dopant_element'].value_counts().index[:6]
    sns.scatterplot(
        data=analysis_df[analysis_df['primary_dopant_element'].isin(top_dopants)],
        x='average_dopant_radius',
        y='pred_Ea',
        hue='primary_dopant_element',
        alpha=0.7
    )
    plt.title('Physics Validation: Activation Energy vs. Dopant Radius')
    plt.xlabel('Average Dopant Ionic Radius (pm)')
    plt.ylabel('Predicted Activation Energy (eV)')

    # --- Plot 2: activation energy vs doping concentration (defect interactions) ---
    plt.subplot(1, 2, 2)
    sns.scatterplot(
        data=analysis_df[analysis_df['primary_dopant_element'].isin(top_dopants)],
        x='total_dopant_fraction',
        y='pred_Ea',
        hue='primary_dopant_element',
        alpha=0.7
    )
    plt.title('Physics Validation: Activation Energy vs. Dopant Concentration')
    plt.xlabel('Total Dopant Molar Fraction')
    plt.ylabel('Predicted Activation Energy (eV)')

    plt.tight_layout()
    plt.savefig(path_config.EA_VS_STRUCTURE_AND_DOPING_IMAGE_PATH)
    print(f"   -> Saved physics analysis plots to '{path_config.EA_VS_STRUCTURE_AND_DOPING_IMAGE_PATH}'")

    return analysis_df

# ================= 5. Module 3: virtual screening =================
def virtual_screening(model, pipeline, train_df):
    print("\n>>> [Module 3] Running Virtual Screening for New Materials...")

    # Define search space: dopant elements and fractions
    dopants = ['Sc', 'Y', 'Yb', 'Gd']
    fractions = [0.06, 0.08, 0.10, 0.12, 0.14, 0.16]

    # Define ionic radii map (Shannon radii, VI coord, 3+)
    radius_map = {'Sc': 74.5, 'Y': 90.0, 'Yb': 86.8, 'Gd': 93.8} # units: pm (approx.)

    virtual_samples = []

    # Use one training row as a template to ensure all non-key columns have default values
    base_row = train_df.iloc[0].copy()

    for dopant in dopants:
        for frac in fractions:
            row = base_row.copy()
            # Set virtual sample ID
            row['sample_id'] = f"Virtual_{dopant}_{frac:.2f}"

            # Set core chemical features
            row['primary_dopant_element'] = dopant
            row['total_dopant_fraction'] = frac
            row['average_dopant_radius'] = radius_map.get(dopant, 90.0) # safe default
            row['average_dopant_valence'] = 3.0 # assume all are +3 rare-earth dopants
            row['number_of_dopants'] = 1

            # Set fixed process parameters (standardized process)
            row['maximum_sintering_temperature'] = 1550
            row['total_sintering_duration'] = 10
            row['synthesis_method'] = 'Solid State Reaction'

            # Set fixed test conditions (e.g., 800°C)
            target_temp_c = 800
            row['operating_temperature'] = target_temp_c
            row['temperature_kelvin'] = target_temp_c + 273.15

            # Must fill text features, otherwise TfidfVectorizer will error
            row['material_source_and_purity'] = "Virtual Screening Generated Sample High Purity"

            virtual_samples.append(row)

    virtual_df = pd.DataFrame(virtual_samples)

    # Feature transform
    # Note: the pipeline handles text, categorical encoding, and numeric scaling
    X_virtual = pipeline.transform(virtual_df)

    X_v_tensor = torch.FloatTensor(X_virtual).to(DEVICE)
    T_v_tensor = torch.FloatTensor(virtual_df['temperature_kelvin'].values).view(-1, 1).to(DEVICE)

    # Predict
    model.eval()
    with torch.no_grad():
        preds, Ea, logA = model(X_v_tensor, T_v_tensor)

    virtual_df['pred_log_sigma'] = preds.cpu().numpy()
    virtual_df['pred_Ea'] = Ea.cpu().numpy()
    virtual_df['pred_sigma'] = 10 ** virtual_df['pred_log_sigma'] # convert back to conductivity

    # Select top candidates
    top_candidates = virtual_df.sort_values('pred_log_sigma', ascending=False).head(5)

    print(f"\n>>> Top 5 Predicted Candidates (at 800°C):")
    cols_to_show = ['sample_id', 'primary_dopant_element', 'total_dopant_fraction', 'pred_Ea', 'pred_sigma']
    print(top_candidates[cols_to_show])

    # Save results
    top_candidates.to_csv(path_config.VIRTUAL_SCREENING_RESULTS_CSV, index=False)
    print(f"   -> Saved screening results to '{path_config.VIRTUAL_SCREENING_RESULTS_CSV}'")

    return top_candidates

# ================= 6. Main =================
def main():
    # 1. Prepare data
    train_df, test_df, X_train, X_test, y_train, y_test, pipeline = get_data_and_pipeline()

    # -------------------------------------------------------------
    # [New] Prepare standardized temperatures for Baseline DNN comparison
    # Must exactly replicate the logic in 03a_train_baseline_model.py:
    # fit on Train, transform on Test
    # -------------------------------------------------------------
    print(">>> [Setup] Scaling temperature for Baseline DNN comparison...")
    t_scaler = StandardScaler()
    t_scaler.fit(train_df[['temperature_kelvin']].values)
    T_test_scaled = t_scaler.transform(test_df[['temperature_kelvin']].values)
    T_test_scaled_tensor = torch.FloatTensor(T_test_scaled).to(DEVICE)
    # -------------------------------------------------------------

    # 2. Run traditional ML benchmarks (RF & XGB)
    bench_results = run_benchmark(X_train, y_train, X_test, y_test)

    # -------------------------------------------------------------
    # [New] Evaluate Baseline DNN (Standard DNN)
    # -------------------------------------------------------------
    print("\n>>> [Benchmark] Evaluating Standard DNN (Baseline)...")
    input_dim = X_train.shape[1]
    baseline_model = StandardDNN(input_dim).to(DEVICE)

    # Check whether the model exists
    if os.path.exists(path_config.BASELINE_MODEL_PATH):
        baseline_model.load_state_dict(torch.load(path_config.BASELINE_MODEL_PATH, map_location=DEVICE))
        baseline_model.eval()

        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)

        with torch.no_grad():
            # StandardDNN.forward accepts (x, t_scaled)
            baseline_preds = baseline_model(X_test_tensor, T_test_scaled_tensor)

        baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds.cpu().numpy()))
        baseline_r2 = r2_score(y_test, baseline_preds.cpu().numpy())

        bench_results['Standard DNN'] = {'RMSE': baseline_rmse, 'R2': baseline_r2}
    else:
        print(f"Warning: Baseline model file not found at {path_config.BASELINE_MODEL_PATH}")
        print("Please run '03a_train_baseline_model.py' first to generate the baseline.")
        bench_results['Standard DNN'] = {'RMSE': np.nan, 'R2': np.nan}


    # 3. Load and evaluate PIML model (ours)
    print(f"\n>>> [Benchmark] Evaluating PIML Model (Ours)...")
    piml_model = PhysicsInformedNet(input_dim).to(DEVICE)

    if os.path.exists(path_config.BEST_PIML_MODEL_PATH):
        piml_model.load_state_dict(torch.load(path_config.BEST_PIML_MODEL_PATH, map_location=DEVICE))

        # 4. Evaluate PIML model (compare to benchmarks)
        piml_model.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        T_test_tensor = torch.FloatTensor(test_df['temperature_kelvin'].values).view(-1, 1).to(DEVICE)

        with torch.no_grad():
            piml_preds, _, _ = piml_model(X_test_tensor, T_test_tensor)

        piml_rmse = np.sqrt(mean_squared_error(y_test, piml_preds.cpu().numpy()))
        piml_r2 = r2_score(y_test, piml_preds.cpu().numpy())

        bench_results['PIML (Ours)'] = {'RMSE': piml_rmse, 'R2': piml_r2}
    else:
        print(f"Error: PIML Model file not found at {path_config.BEST_PIML_MODEL_PATH}")
        print("Please run '01_train_physics_model.py' first.")

    # Print final comparison results
    print("\n========================================")
    print("      Model Comparison Results          ")
    print("========================================")
    df_results = pd.DataFrame(bench_results).T
    # Format output to 4 decimals
    print(df_results.applymap(lambda x: f"{x:.4f}"))

    # -------------------------------------------------------------
    # [Key] Save comparison table as CSV (per your config)
    # -------------------------------------------------------------
    df_results.to_csv(path_config.FINAL_METRICS_COMPARISON_CSV)
    print(f"\n>>> Metrics saved to {path_config.FINAL_METRICS_COMPARISON_CSV}")

    # 5. Run physics analysis (PIML only)
    if os.path.exists(path_config.BEST_PIML_MODEL_PATH):
        analyze_physics(piml_model, X_test, test_df)

        # 6. Run virtual screening (PIML only)
        virtual_screening(piml_model, pipeline, train_df)

if __name__ == "__main__":
    main()

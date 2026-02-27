import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader

from config import path_config
from etl.material_data_processor import MaterialDataProcessor
from features.preprocessor import build_feature_pipeline
from custom_datasets.conductivity_dataset import ConductivityDataset
from models.baseline_net import StandardDNN

# --- Local parameters ---
SEED = 42
BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(SEED)

    # ==========================================
    # 1. Directory initialization (moved to 00_init_dir.py)
    # ==========================================


    # ==========================================
    # 2. Data loading & preprocessing
    # ==========================================
    # Reuse the unified ETL logic
    processor = MaterialDataProcessor()
    df = processor.load_and_preprocess_data_for_training_piml()

    target_col = 'log_conductivity'
    temperature_col = 'temperature_kelvin'

    # Split dataset
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)

    # ==========================================
    # 3. Feature engineering
    # ==========================================

    # A. Material features (X) - reuse pipeline
    print(">>> Fitting feature pipeline...")
    pipeline = build_feature_pipeline()
    X_train = pipeline.fit_transform(train_df)
    X_val = pipeline.transform(val_df)

    input_dim = X_train.shape[1]

    # B. Temperature feature (T) - baseline-specific step
    # Pure DNNs are sensitive to input scale; standardize temperature (z-score).
    print(">>> Scaling temperature for DNN baseline...")
    t_scaler = StandardScaler()
    T_train_scaled = t_scaler.fit_transform(train_df[[temperature_col]].values)
    T_val_scaled = t_scaler.transform(val_df[[temperature_col]].values)

    # C. Build Dataset
    # Although the Dataset field is named 'temps', we pass the standardized T_scaled.
    train_dataset = ConductivityDataset(X_train, T_train_scaled, train_df[target_col].values)
    val_dataset = ConductivityDataset(X_val, T_val_scaled, val_df[target_col].values)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ==========================================
    # 4. Model initialization
    # ==========================================
    model = StandardDNN(input_dim).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # ==========================================
    # 5. Training loop
    # ==========================================
    best_val_loss = float('inf')

    print(f">>> Starting Baseline Training on {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, T_batch, y_batch in train_loader:
            X_batch, T_batch, y_batch = X_batch.to(DEVICE), T_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            preds = model(X_batch, T_batch) # Forward: X + Scaled_T
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, T_batch, y_batch in val_loader:
                X_batch, T_batch, y_batch = X_batch.to(DEVICE), T_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(X_batch, T_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), path_config.BASELINE_MODEL_PATH)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{EPOCHS} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

    print(f">>> Best baseline model saved to: {path_config.BASELINE_MODEL_PATH}")

    # ==========================================
    # 6. Evaluation & visualization
    # ==========================================
    model.load_state_dict(torch.load(path_config.BASELINE_MODEL_PATH))
    model.eval()

    X_val_tensor = torch.FloatTensor(X_val).to(DEVICE)
    T_val_scaled_tensor = torch.FloatTensor(T_val_scaled).to(DEVICE)

    with torch.no_grad():
        preds = model(X_val_tensor, T_val_scaled_tensor)

    val_df = val_df.copy()
    val_df['predicted_log_sigma'] = preds.cpu().numpy()

    # Metrics
    rmse = np.sqrt(mean_squared_error(val_df[target_col], val_df['predicted_log_sigma']))
    r2 = r2_score(val_df[target_col], val_df['predicted_log_sigma'])
    print(f"\n>>> Baseline DNN Final Results:")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    R2 Score: {r2:.4f}")

    # Plot
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=target_col, y='predicted_log_sigma', data=val_df, alpha=0.6, color='gray')

    # Reference line
    min_val = min(val_df[target_col].min(), val_df['predicted_log_sigma'].min())
    max_val = max(val_df[target_col].max(), val_df['predicted_log_sigma'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

    plt.title(f"Baseline DNN (Pure Data-Driven)\nRMSE={rmse:.3f}, R2={r2:.3f}")
    plt.xlabel("Actual Log10 Conductivity")
    plt.ylabel("Predicted Log10 Conductivity")
    plt.legend()
    plt.tight_layout()

    plt.savefig(path_config.BASELINE_DNN_RESULT_IMAGE_PATH)
    print(f">>> Result plot saved to {path_config.BASELINE_DNN_RESULT_IMAGE_PATH}")

    # Save prediction CSV (for ablation comparison with PIML in the paper)
    comparison_df = val_df[['sample_id', temperature_col, target_col, 'predicted_log_sigma']].copy()
    comparison_df.rename(columns={'predicted_log_sigma': 'pred_baseline_dnn'}, inplace=True)
    comparison_df.to_csv(path_config.COMPARISON_BASELINE_DNN_CSV, index=False)
    print(f">>> Prediction CSV saved to {path_config.COMPARISON_BASELINE_DNN_CSV}")

if __name__ == "__main__":
    main()

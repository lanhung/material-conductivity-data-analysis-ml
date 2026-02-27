import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import warnings

from config import path_config
from etl.material_data_processor import MaterialDataProcessor
from features.preprocessor import build_feature_pipeline
from models.piml_net import PhysicsInformedNet

# Suppress warnings
warnings.filterwarnings('ignore')

# ================= 1. Config & parameters =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Set random seeds
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)





# ================= 2. Helper training function =================
def train_experiment_model(X_train, y_train, T_train, epochs=60):
    """
    Helper function to quickly train a model for analysis experiments.
    """
    input_dim = X_train.shape[1]
    model = PhysicsInformedNet(input_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()

    X_t = torch.FloatTensor(X_train).to(DEVICE)
    y_t = torch.FloatTensor(y_train).view(-1, 1).to(DEVICE)
    T_t = torch.FloatTensor(T_train).view(-1, 1).to(DEVICE)

    print(f"   Training analysis model for {epochs} epochs on {len(X_train)} samples...")

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        # forward returns: log_sigma_pred, Ea, log_A
        preds, _, _ = model(X_t, T_t)
        loss = criterion(preds, y_t)
        loss.backward()
        optimizer.step()

    return model

# ================= 3. Main experiments =================
def run_experiments():
    # --- A. Data loading (reuse unified ETL) ---
    processor = MaterialDataProcessor()
    df = processor.load_and_preprocess_data_for_training_piml()

    # Key column mapping (based on MaterialDataProcessor SQL output)
    target_col = 'log_conductivity'
    temperature_col = 'temperature_kelvin'
    dopant_col = 'primary_dopant_element'
    synthesis_col = 'synthesis_method'

    # --- B. Feature engineering (reuse unified pipeline) ---
    pipeline = build_feature_pipeline()
    # Note: fit_transform returns a numpy array
    X_full = pipeline.fit_transform(df)

    # Get feature names (for feature importance analysis)
    # Get numeric and categorical feature names
    feat_num = pipeline.named_transformers_['num'].get_feature_names_out().tolist()
    feat_cat = pipeline.named_transformers_['cat'].get_feature_names_out().tolist()
    # Text features go through PCA/SVD; name them manually
    feat_names = feat_num + feat_cat + [f"text_svd_{i}" for i in range(16)] # SVD components=16 in preprocessor.py

    # Prepare tensors for reuse
    X_tensor = torch.FloatTensor(X_full).to(DEVICE)
    T_tensor = torch.FloatTensor(df[temperature_col].values).view(-1, 1).to(DEVICE)
    y_tensor = torch.FloatTensor(df[target_col].values).view(-1, 1).to(DEVICE)

    # =========================================================
    # --- Experiment 1: latent space visualization (manifold learning) ---
    # =========================================================
    print("\n[Experiment 1] Visualizing Latent Chemical Space (t-SNE)...")

    # 1. Train model on full dataset
    model_full = train_experiment_model(X_full, df[target_col].values, df[temperature_col].values, epochs=60)
    model_full.eval()

    # 2. Extract latent features
    # Note: piml_net.py's forward does not return hidden, so we call model.encoder directly.
    with torch.no_grad():
        latent = model_full.encoder(X_tensor)
        # Also retrieve physical parameters for later use
        _, base_Ea, _ = model_full(X_tensor, T_tensor)

    # 3. t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_2d = tsne.fit_transform(latent.cpu().numpy())

    df['tsne_1'] = latent_2d[:, 0]
    df['tsne_2'] = latent_2d[:, 1]

    # 4. Plot
    plt.figure(figsize=(10, 6))
    # Only show top 6 dopants to avoid a cluttered legend
    top_dopants = df[dopant_col].value_counts().index[:6]
    sns.scatterplot(data=df[df[dopant_col].isin(top_dopants)],
                    x='tsne_1', y='tsne_2',
                    hue=dopant_col, style=synthesis_col, alpha=0.8)
    plt.title("Learned Chemical Space (t-SNE Visualization)")

    plt.savefig(path_config.LATENT_SPACE_IMAGE_PATH)
    print(f"   -> Saved '{path_config.LATENT_SPACE_IMAGE_PATH}'")

    # =========================================================
    # --- Experiment 2: attribution analysis (feature importance for Ea) ---
    # =========================================================
    print("\n[Experiment 2] Analyzing What Drives Activation Energy (Permutation Importance)...")

    base_Ea_np = base_Ea.cpu().numpy().flatten()
    importances = {}

    # Permutation importance
    for i, name in enumerate(feat_names):
        if i >= X_full.shape[1]: break # guard against feature-name index overflow

        X_perm = X_full.copy()
        np.random.shuffle(X_perm[:, i]) # shuffle one feature column

        with torch.no_grad():
            X_perm_tensor = torch.FloatTensor(X_perm).to(DEVICE)
            # Get new hidden via encoder
            hidden_perm = model_full.encoder(X_perm_tensor)
            # Get new Ea via head_Ea
            perm_Ea = model_full.head_Ea(hidden_perm)

        # Compute mean absolute deviation in Ea
        imp_score = np.mean(np.abs(perm_Ea.cpu().numpy().flatten() - base_Ea_np))
        importances[name] = imp_score

    # Plot
    imp_df = pd.Series(importances).sort_values(ascending=False).head(12)
    plt.figure(figsize=(10, 5))
    imp_df.plot(kind='barh', color='#2ca02c')
    plt.title("Feature Importance for Activation Energy (Physical Interpretation)")
    plt.xlabel("Impact on Ea (eV)")
    plt.tight_layout()

    plt.savefig(path_config.PAPER_FEATURE_IMPORTANCE_EA_IMAGE_PATH)
    print(f"   -> Saved '{path_config.PAPER_FEATURE_IMPORTANCE_EA_IMAGE_PATH}'")

    # =========================================================
    # --- Experiment 3: unseen-material discovery test (leave-one-dopant-out) ---
    # =========================================================
    print("\n[Experiment 3] Zero-Shot Discovery: Leave-One-Dopant-Out...")

    # Target test element: Sc (scandium)
    target_element = 'Sc'

    mask_train = df[dopant_col] != target_element
    mask_test = df[dopant_col] == target_element

    # Check whether the dataset contains this element
    if mask_test.sum() == 0:
        print(f"   Warning: No {target_element} samples found. Skipping specific test.")
        # Fallback: use the second most common element as the test set
        target_element = df[dopant_col].value_counts().index[1]
        print(f"   Fallback: Testing Leave-{target_element}-Out instead.")
        mask_train = df[dopant_col] != target_element
        mask_test = df[dopant_col] == target_element

    X_tr = X_full[mask_train]
    y_tr = df[mask_train][target_col].values
    T_tr = df[mask_train][temperature_col].values

    X_te = X_full[mask_test]
    y_te = df[mask_test][target_col].values
    T_te = df[mask_test][temperature_col].values

    # Retrain model (excluding the target element)
    model_lodo = train_experiment_model(X_tr, y_tr, T_tr, epochs=80)
    model_lodo.eval()

    with torch.no_grad():
        preds_sc, _, _ = model_lodo(torch.FloatTensor(X_te).to(DEVICE),
                                    torch.FloatTensor(T_te).view(-1, 1).to(DEVICE))

    preds_sc = preds_sc.cpu().numpy().flatten()
    rmse = np.sqrt(np.mean((preds_sc - y_te)**2))

    # Naive baseline: predict all test samples with the train mean
    naive_pred = np.full_like(y_te, y_tr.mean())
    naive_rmse = np.sqrt(np.mean((naive_pred - y_te)**2))

    # R² score
    ss_res = np.sum((preds_sc - y_te)**2)
    ss_tot = np.sum((y_te - y_te.mean())**2)
    r2 = 1 - ss_res / ss_tot

    plt.figure(figsize=(6, 6))
    plt.scatter(y_te, preds_sc, color='purple', alpha=0.6, label=f'Test Samples ({target_element})')
    # Plot perfect-prediction line
    min_val = min(y_te.min(), preds_sc.min())
    max_val = max(y_te.max(), preds_sc.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
    # Plot naive baseline (train mean)
    plt.axhline(y=y_tr.mean(), color='gray', linestyle=':', alpha=0.8, label=f'Naive Baseline (train mean={y_tr.mean():.2f})')

    plt.title(
        f"Generalization to Unseen Element ({target_element})\n"
        f"RMSE: {rmse:.3f}  |  Naive RMSE: {naive_rmse:.3f}  |  R²: {r2:.3f}"
    )
    plt.xlabel("Actual log(sigma)")
    plt.ylabel("Predicted log(sigma)")
    plt.legend()

    save_path_3 = path_config.IMAGE_DIR / f"paper_lodo_{target_element}.png"
    plt.savefig(save_path_3)
    print(f"   -> Saved '{save_path_3}'.")
    print(f"      RMSE={rmse:.4f}  |  Naive RMSE={naive_rmse:.4f}  |  R²={r2:.4f}")

if __name__ == "__main__":
    run_experiments()

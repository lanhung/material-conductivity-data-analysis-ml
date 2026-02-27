import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import sys


from config import path_config
from etl.material_data_processor import MaterialDataProcessor
from features.preprocessor import build_feature_pipeline
from models.piml_net import PhysicsInformedNet

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


# ================= 2. Core utility: MC Dropout uncertainty prediction =================
def predict_with_uncertainty(model, X, T_K, n_iter=50):
    """
    Keep Dropout enabled during inference (Monte Carlo Dropout) and run multiple forward passes
    to estimate the predictive mean and uncertainty (std dev).
    """
    # Key: set the model to train mode so Dropout layers stay active.
    model.train()

    X_t = torch.FloatTensor(X).to(DEVICE)
    T_t = torch.FloatTensor(T_K).view(-1, 1).to(DEVICE)

    preds_list = []
    with torch.no_grad():
        for _ in range(n_iter):
            # forward returns: log_sigma, Ea, logA
            preds, _, _ = model(X_t, T_t)
            preds_list.append(preds.cpu().numpy())

    # Shape: (n_iter, n_samples, 1)
    preds_arr = np.array(preds_list)

    # Use the mean as the final prediction and the standard deviation as uncertainty (epistemic).
    mean_pred = preds_arr.mean(axis=0).flatten()
    std_pred = preds_arr.std(axis=0).flatten()

    return mean_pred, std_pred

# ================= 3. Data preparation =================
def get_data_ready():
    print(">>> [Setup] Loading and processing data...")
    processor = MaterialDataProcessor()
    df = processor.load_and_preprocess_data_for_training_piml()

    pipeline = build_feature_pipeline()
    X = pipeline.fit_transform(df)

    return df, X

# ================= 4. Experiment 4: uncertainty calibration (UQ) =================
def run_uq_experiment(df, X):
    """
    Train a model and check whether it can estimate error ranges correctly.
    Ideally, the predicted uncertainty (error bars) should cover the true values.
    """
    print("\n>>> [Module 1] Running Uncertainty Quantification (UQ)...")

    # Split data
    X_train, X_test, y_train, y_test, T_train, T_test = train_test_split(
        X, df['log_conductivity'].values, df['temperature_kelvin'].values,
        test_size=0.2, random_state=SEED
    )

    # Quickly train an ad-hoc model (for UQ demo)
    input_dim = X.shape[1]
    model = PhysicsInformedNet(input_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()

    X_tr_t = torch.FloatTensor(X_train).to(DEVICE)
    y_tr_t = torch.FloatTensor(y_train).view(-1, 1).to(DEVICE)
    T_tr_t = torch.FloatTensor(T_train).view(-1, 1).to(DEVICE)

    print("    Training UQ probe model...")
    for ep in range(150): # simple training loop
        model.train()
        optimizer.zero_grad()
        preds, _, _ = model(X_tr_t, T_tr_t)
        loss = criterion(preds, y_tr_t)
        loss.backward()
        optimizer.step()

    # Predict with MC Dropout
    mu, sigma = predict_with_uncertainty(model, X_test, T_test, n_iter=100)

    # --- Plot ---
    plt.figure(figsize=(7, 7))

    # Randomly sample 50 points for visualization to avoid overcrowding.
    if len(y_test) > 50:
        indices = np.random.choice(len(y_test), 50, replace=False)
    else:
        indices = np.arange(len(y_test))

    # Draw scatter with error bars (95% CI = 1.96 * std)
    plt.errorbar(
        y_test[indices],
        mu[indices],
        yerr=1.96 * sigma[indices],
        fmt='o', ecolor='gray', alpha=0.6, capsize=3,
        label='95% Confidence Interval'
    )

    # Draw ideal prediction line (y=x)
    min_val, max_val = min(y_test), max(y_test)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')

    plt.title("Uncertainty Quantification (Monte Carlo Dropout)")
    plt.xlabel("Actual Log Conductivity")
    plt.ylabel("Predicted Log Conductivity")
    plt.legend()
    plt.tight_layout()

    save_path = path_config.UQ_CALIBRATION_IMAGE_PATH
    plt.savefig(save_path)
    print(f"    -> Saved calibration plot to '{save_path}'")

# ================= 5. Experiment 5: active learning simulation =================
def run_active_learning_simulation(df, X):
    """
    Simulate an "AI scientist": compare random trials vs AI-guided (active learning) discovery speed for high-performance materials.
    """
    print("\n>>> [Module 2] Running Active Learning Simulation (AI Scientist)...")

    # Simulation settings
    n_samples = len(df)
    n_initial = int(n_samples * 0.05) # start with only 5% of the data
    n_step = 5   # 5 samples per batch
    n_rounds = 15 # run 15 rounds

    # Index pools
    indices = np.random.permutation(n_samples)
    initial_idx = indices[:n_initial]
    pool_idx = indices[n_initial:]

    # Strategies
    # 1. Random: blind exploration
    # 2. Greedy (AI-guided): exploitation; always test what the model thinks is best
    strategies = ['Random', 'Greedy (AI-Guided)']
    results = {s: [] for s in strategies}

    # Helper: train quickly and select samples
    def train_and_select(train_idx, pool_idx, strategy):
        # Prepare current training set
        X_tr = X[train_idx]
        y_tr = df.iloc[train_idx]['log_conductivity'].values
        T_tr = df.iloc[train_idx]['temperature_kelvin'].values

        # Prepare candidate pool
        X_pool = X[pool_idx]
        T_pool = df.iloc[pool_idx]['temperature_kelvin'].values

        # 1. Train a model from scratch (simulate real scenario: update after each new batch)
        model = PhysicsInformedNet(X.shape[1]).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=0.005)
        loss_fn = nn.MSELoss()

        X_t = torch.FloatTensor(X_tr).to(DEVICE)
        y_t = torch.FloatTensor(y_tr).view(-1, 1).to(DEVICE)
        T_t = torch.FloatTensor(T_tr).view(-1, 1).to(DEVICE)

        model.train()
        for _ in range(60): # quick training for 60 epochs
            opt.zero_grad()
            pred, _, _ = model(X_t, T_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            opt.step()

        # 2. Predict on candidate pool
        # Use MC Dropout to get mean (mu) and uncertainty (sigma)
        # Note: greedy mainly uses mu; an exploration strategy would use sigma.
        mu, sigma = predict_with_uncertainty(model, X_pool, T_pool, n_iter=20)

        # 3. Select samples
        if strategy == 'Random':
            selected_local_idx = np.random.choice(len(pool_idx), n_step, replace=False)
        elif strategy == 'Greedy (AI-Guided)':
            # Pick the top-N highest predicted conductivity
            selected_local_idx = np.argsort(mu)[::-1][:n_step]

        # 4. Track the current "best material" found (max true conductivity)
        # Simulate: what's the best value among already-tested samples (train_idx)?
        max_found = np.max(y_tr)

        return selected_local_idx, max_found

    # Run simulation loop
    for strategy in strategies:
        print(f"    Running strategy: {strategy}...")
        curr_train = initial_idx.copy()
        curr_pool = pool_idx.copy()

        history = []

        for r in range(n_rounds):
            selected_local, max_val = train_and_select(curr_train, curr_pool, strategy)
            history.append(max_val)

            # Update pools: move selected samples from pool to train
            # Note on index mapping: selected_local is relative to curr_pool
            selected_global = curr_pool[selected_local]

            curr_train = np.concatenate([curr_train, selected_global])
            curr_pool = np.delete(curr_pool, selected_local)

            # Simple progress indicator
            print(f"      Round {r+1}/{n_rounds} | Best Found: {max_val:.4f}", end='\r')
        print(f"      Strategy {strategy} completed.             ")

        results[strategy] = history

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    for name, hist in results.items():
        plt.plot(range(len(hist)), hist, marker='o', label=name, linewidth=2)

    plt.title("Accelerated Discovery: AI vs Random Sampling")
    plt.xlabel("Experimental Batches (Iterations)")
    plt.ylabel("Max Conductivity Found (log S/cm)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = path_config.ACTIVE_LEARNING_IMAGE_PATH
    plt.savefig(save_path)
    print(f"    -> Saved active learning plot to '{save_path}'")

if __name__ == "__main__":
    # 1. Prepare data
    df, X = get_data_ready()

    # 2. Run uncertainty experiment
    run_uq_experiment(df, X)

    # 3. Run active learning simulation
    run_active_learning_simulation(df, X)

    print("\n>>> All Lab Application experiments completed.")

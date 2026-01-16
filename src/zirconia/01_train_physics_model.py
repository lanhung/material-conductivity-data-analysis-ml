import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

# --- 1. 模块化导入 ---
from etl.material_data_processor import MaterialDataProcessor
from features.preprocessor import build_feature_pipeline
from datasets.conductivity_dataset import ConductivityDataset
from models.piml_net import PhysicsInformedNet


from config import path_config

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
    # 1. 目录初始化，提取到了00_init_dir.py
    # ==========================================


    print(f">>> [Setup] Output directories ensure created at: {path_config.RESULTS_DIR}")

    # ==========================================
    # 2. 数据处理与加载
    # ==========================================
    processor = MaterialDataProcessor()
    df = processor.load_and_preprocess_data_for_training_piml()

    # 简单的列名映射
    target_col = 'log_conductivity'
    temperature_col = 'temperature_kelvin'

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)

    pipeline = build_feature_pipeline()
    X_train = pipeline.fit_transform(train_df)
    X_val = pipeline.transform(val_df)

    input_dim = X_train.shape[1]

    train_dataset = ConductivityDataset(X_train, train_df[temperature_col].values, train_df[target_col].values)
    val_dataset = ConductivityDataset(X_val, val_df[temperature_col].values, val_df[target_col].values)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ==========================================
    # 3. 训练流程
    # ==========================================
    model = PhysicsInformedNet(input_dim).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    print(">>> Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, T_batch, y_batch in train_loader:
            X_batch, T_batch, y_batch = X_batch.to(DEVICE), T_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            preds, _, _ = model(X_batch, T_batch)
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
                preds, _, _ = model(X_batch, T_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # --- 保存模型 (使用配置路径) ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), path_config.BEST_PIML_MODEL_PATH)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{EPOCHS} | Val Loss: {avg_val_loss:.4f}")

    print(f">>> Best model saved to: {path_config.BEST_PIML_MODEL_PATH}")

    # ==========================================
    # 4. 评估与绘图
    # ==========================================
    # 加载模型 (使用配置路径)
    model.load_state_dict(torch.load(path_config.BEST_PIML_MODEL_PATH))
    model.eval()

    X_val_tensor = torch.FloatTensor(X_val).to(DEVICE)
    T_val_tensor = torch.FloatTensor(val_df[temperature_col].values).view(-1, 1).to(DEVICE)

    with torch.no_grad():
        preds, Ea_pred, logA_pred = model(X_val_tensor, T_val_tensor)

    val_df = val_df.copy()
    val_df['predicted_log_sigma'] = preds.cpu().numpy()
    val_df['predicted_Ea'] = Ea_pred.cpu().numpy()

    # 绘图
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x=target_col, y='predicted_log_sigma', data=val_df, alpha=0.6)
    min_val = min(val_df[target_col].min(), val_df['predicted_log_sigma'].min())
    max_val = max(val_df[target_col].max(), val_df['predicted_log_sigma'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title("Prediction vs Actual")

    plt.subplot(1, 2, 2)
    sns.histplot(val_df['predicted_Ea'], kde=True, color='purple')
    plt.title("Predicted Activation Energy (Ea)")

    plt.tight_layout()

    # --- 保存图片 (使用配置路径) ---
    plt.savefig(path_config.PIML_PREDICTION_EA_DISTANCE_IMAGE_PATH)
    print(f">>> Results image saved to: {path_config.PIML_PREDICTION_EA_DISTANCE_IMAGE_PATH}")

    rmse = np.sqrt(mean_squared_error(val_df[target_col], val_df['predicted_log_sigma']))
    print(f"Final RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main()
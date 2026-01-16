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

# 抑制警告
warnings.filterwarnings('ignore')

# ================= 1. 配置与参数 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# 统一设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)





# ================= 2. 辅助训练函数 =================
def train_experiment_model(X_train, y_train, T_train, epochs=60):
    """
    为分析实验快速训练一个模型的辅助函数。
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
        # forward 返回: log_sigma_pred, Ea, log_A
        preds, _, _ = model(X_t, T_t)
        loss = criterion(preds, y_t)
        loss.backward()
        optimizer.step()

    return model

# ================= 3. 主实验逻辑 =================
def run_experiments():
    # --- A. 数据加载 (使用统一的 ETL) ---
    processor = MaterialDataProcessor()
    df = processor.load_and_preprocess_data_for_training_piml()

    # 关键列名映射 (基于 MaterialDataProcessor 的 SQL 输出)
    target_col = 'log_conductivity'
    temperature_col = 'temperature_kelvin'
    dopant_col = 'primary_dopant_element'
    synthesis_col = 'synthesis_method'

    # --- B. 特征工程 (使用统一的 Pipeline) ---
    pipeline = build_feature_pipeline()
    # 注意：fit_transform 返回的是 numpy array
    X_full = pipeline.fit_transform(df)

    # 获取特征名称（用于特征重要性分析）
    # 获取数值和类别特征名
    feat_num = pipeline.named_transformers_['num'].get_feature_names_out().tolist()
    feat_cat = pipeline.named_transformers_['cat'].get_feature_names_out().tolist()
    # 文本特征经过了 PCA/SVD，手动命名
    feat_names = feat_num + feat_cat + [f"text_svd_{i}" for i in range(16)] # SVD components=16 in preprocessor.py

    # 准备 Tensor 数据供后续复用
    X_tensor = torch.FloatTensor(X_full).to(DEVICE)
    T_tensor = torch.FloatTensor(df[temperature_col].values).view(-1, 1).to(DEVICE)
    y_tensor = torch.FloatTensor(df[target_col].values).view(-1, 1).to(DEVICE)

    # =========================================================
    # --- 实验 1: 隐空间可视化 (Manifold Learning) ---
    # =========================================================
    print("\n[Experiment 1] Visualizing Latent Chemical Space (t-SNE)...")

    # 1. 训练全量模型
    model_full = train_experiment_model(X_full, df[target_col].values, df[temperature_col].values, epochs=60)
    model_full.eval()

    # 2. 提取隐层特征
    # 注意：piml_net.py 的 forward 不返回 hidden，所以我们直接调用 model.encoder
    with torch.no_grad():
        latent = model_full.encoder(X_tensor)
        # 同时获取物理参数供后续使用
        _, base_Ea, _ = model_full(X_tensor, T_tensor)

    # 3. t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_2d = tsne.fit_transform(latent.cpu().numpy())

    df['tsne_1'] = latent_2d[:, 0]
    df['tsne_2'] = latent_2d[:, 1]

    # 4. 绘图
    plt.figure(figsize=(10, 6))
    # 只取最常见的 top 6 掺杂元素以避免图例混乱
    top_dopants = df[dopant_col].value_counts().index[:6]
    sns.scatterplot(data=df[df[dopant_col].isin(top_dopants)],
                    x='tsne_1', y='tsne_2',
                    hue=dopant_col, style=synthesis_col, alpha=0.8)
    plt.title("Learned Chemical Space (t-SNE Visualization)")

    plt.savefig(path_config.LATENT_SPACE_IMAGE_PATH)
    print(f"   -> Saved '{path_config.LATENT_SPACE_IMAGE_PATH}'")

    # =========================================================
    # --- 实验 2: 物理参数归因分析 (Feature Importance for Ea) ---
    # =========================================================
    print("\n[Experiment 2] Analyzing What Drives Activation Energy (Permutation Importance)...")

    base_Ea_np = base_Ea.cpu().numpy().flatten()
    importances = {}

    # 排列重要性分析 (Permutation Importance)
    for i, name in enumerate(feat_names):
        if i >= X_full.shape[1]: break # 防止特征名索引越界

        X_perm = X_full.copy()
        np.random.shuffle(X_perm[:, i]) # 打乱某一列特征

        with torch.no_grad():
            X_perm_tensor = torch.FloatTensor(X_perm).to(DEVICE)
            # 通过 encoder 获取新的 hidden
            hidden_perm = model_full.encoder(X_perm_tensor)
            # 通过 head_Ea 获取新的 Ea
            perm_Ea = model_full.head_Ea(hidden_perm)

        # 计算 Ea 的平均绝对变化幅度 (Mean Absolute Deviation)
        imp_score = np.mean(np.abs(perm_Ea.cpu().numpy().flatten() - base_Ea_np))
        importances[name] = imp_score

    # 绘图
    imp_df = pd.Series(importances).sort_values(ascending=False).head(12)
    plt.figure(figsize=(10, 5))
    imp_df.plot(kind='barh', color='#2ca02c')
    plt.title("Feature Importance for Activation Energy (Physical Interpretation)")
    plt.xlabel("Impact on Ea (eV)")
    plt.tight_layout()

    plt.savefig(path_config.PAPER_FEATURE_IMPORTANCE_EA_IMAGE_PATH)
    print(f"   -> Saved '{path_config.PAPER_FEATURE_IMPORTANCE_EA_IMAGE_PATH}'")

    # =========================================================
    # --- 实验 3: 未知材料发现测试 (Leave-One-Dopant-Out) ---
    # =========================================================
    print("\n[Experiment 3] Zero-Shot Discovery: Leave-One-Dopant-Out...")

    # 目标测试元素：Sc (钪)
    target_element = 'Sc'

    mask_train = df[dopant_col] != target_element
    mask_test = df[dopant_col] == target_element

    # 检查数据集中是否存在该元素
    if mask_test.sum() == 0:
        print(f"   Warning: No {target_element} samples found. Skipping specific test.")
        # Fallback: 使用第二常见的元素作为测试集
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

    # 重新训练模型（不包含目标元素的数据）
    model_lodo = train_experiment_model(X_tr, y_tr, T_tr, epochs=80)
    model_lodo.eval()

    with torch.no_grad():
        preds_sc, _, _ = model_lodo(torch.FloatTensor(X_te).to(DEVICE),
                                    torch.FloatTensor(T_te).view(-1, 1).to(DEVICE))

    preds_sc = preds_sc.cpu().numpy().flatten()
    rmse = np.sqrt(np.mean((preds_sc - y_te)**2))

    plt.figure(figsize=(6, 6))
    plt.scatter(y_te, preds_sc, color='purple', alpha=0.6, label=f'Test Samples ({target_element})')
    # 绘制完美预测线
    min_val = min(y_te.min(), preds_sc.min())
    max_val = max(y_te.max(), preds_sc.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')

    plt.title(f"Generalization to Unseen Element ({target_element})\nRMSE: {rmse:.3f}")
    plt.xlabel("Actual log(sigma)")
    plt.ylabel("Predicted log(sigma)")
    plt.legend()

    save_path_3 = path_config.IMAGE_DIR / f"paper_lodo_{target_element}.png"
    plt.savefig(save_path_3)
    print(f"   -> Saved '{save_path_3}'. RMSE on {target_element}: {rmse:.4f}")

if __name__ == "__main__":
    run_experiments()
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

# 抑制警告
warnings.filterwarnings('ignore')

# ================= 1. 配置与参数 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# ================= 2. 核心工具：MC Dropout 不确定性预测 =================
def predict_with_uncertainty(model, X, T_K, n_iter=50):
    """
    通过在推理阶段保持 Dropout 开启 (Monte Carlo Dropout)，
    进行多次前向传播来估计预测值的均值 (Mean) 和不确定性 (Std Dev)。
    """
    # 关键：设置模型为 train 模式，确保 Dropout 层是活跃的
    model.train()

    X_t = torch.FloatTensor(X).to(DEVICE)
    T_t = torch.FloatTensor(T_K).view(-1, 1).to(DEVICE)

    preds_list = []
    with torch.no_grad():
        for _ in range(n_iter):
            # forward 返回: log_sigma, Ea, logA
            preds, _, _ = model(X_t, T_t)
            preds_list.append(preds.cpu().numpy())

    # Shape: (n_iter, n_samples, 1)
    preds_arr = np.array(preds_list)

    # 计算均值作为最终预测，标准差作为不确定性 (Epistemic Uncertainty)
    mean_pred = preds_arr.mean(axis=0).flatten()
    std_pred = preds_arr.std(axis=0).flatten()

    return mean_pred, std_pred

# ================= 3. 数据准备 =================
def get_data_ready():
    print(">>> [Setup] Loading and processing data...")
    processor = MaterialDataProcessor()
    df = processor.load_and_preprocess_data_for_training_piml()

    pipeline = build_feature_pipeline()
    X = pipeline.fit_transform(df)

    return df, X

# ================= 4. 实验四：不确定性校准 (UQ Calibration) =================
def run_uq_experiment(df, X):
    """
    训练模型并验证其能否正确估计误差范围。
    理想情况下，预测的不确定性范围 (Error Bars) 应覆盖真实值。
    """
    print("\n>>> [Module 1] Running Uncertainty Quantification (UQ)...")

    # 划分数据
    X_train, X_test, y_train, y_test, T_train, T_test = train_test_split(
        X, df['log_conductivity'].values, df['temperature_kelvin'].values,
        test_size=0.2, random_state=SEED
    )

    # 快速训练一个专用模型 (Ad-hoc training for UQ demo)
    input_dim = X.shape[1]
    model = PhysicsInformedNet(input_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()

    X_tr_t = torch.FloatTensor(X_train).to(DEVICE)
    y_tr_t = torch.FloatTensor(y_train).view(-1, 1).to(DEVICE)
    T_tr_t = torch.FloatTensor(T_train).view(-1, 1).to(DEVICE)

    print("    Training UQ probe model...")
    for ep in range(150): # 简单的训练循环
        model.train()
        optimizer.zero_grad()
        preds, _, _ = model(X_tr_t, T_tr_t)
        loss = criterion(preds, y_tr_t)
        loss.backward()
        optimizer.step()

    # 使用 MC Dropout 进行预测
    mu, sigma = predict_with_uncertainty(model, X_test, T_test, n_iter=100)

    # --- 绘图 ---
    plt.figure(figsize=(7, 7))

    # 随机采样 50 个点进行可视化，避免图表过于拥挤
    if len(y_test) > 50:
        indices = np.random.choice(len(y_test), 50, replace=False)
    else:
        indices = np.arange(len(y_test))

    # 绘制带误差棒的散点 (95% 置信区间 = 1.96 * std)
    plt.errorbar(
        y_test[indices],
        mu[indices],
        yerr=1.96 * sigma[indices],
        fmt='o', ecolor='gray', alpha=0.6, capsize=3,
        label='95% Confidence Interval'
    )

    # 绘制理想预测线 (y=x)
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

# ================= 5. 实验五：主动学习模拟 (Active Learning) =================
def run_active_learning_simulation(df, X):
    """
    模拟“AI 科学家”：对比随机尝试与 AI 引导（主动学习）在发现高性能材料速度上的差异。
    """
    print("\n>>> [Module 2] Running Active Learning Simulation (AI Scientist)...")

    # 模拟设置
    n_samples = len(df)
    n_initial = int(n_samples * 0.05) # 初始只有 5% 的数据
    n_step = 5   # 每一轮实验只做 5 个样品
    n_rounds = 15 # 进行 15 轮实验

    # 数据索引池
    indices = np.random.permutation(n_samples)
    initial_idx = indices[:n_initial]
    pool_idx = indices[n_initial:]

    # 定义策略
    # 1. Random: 盲目尝试
    # 2. Greedy (AI-Guided): 利用 (Exploitation)，总是测试模型认为最好的材料
    strategies = ['Random', 'Greedy (AI-Guided)']
    results = {s: [] for s in strategies}

    # 内部辅助函数：快速训练并选择样本
    def train_and_select(train_idx, pool_idx, strategy):
        # 准备当前训练集
        X_tr = X[train_idx]
        y_tr = df.iloc[train_idx]['log_conductivity'].values
        T_tr = df.iloc[train_idx]['temperature_kelvin'].values

        # 准备候选池 (Pool)
        X_pool = X[pool_idx]
        T_pool = df.iloc[pool_idx]['temperature_kelvin'].values

        # 1. 从头训练一个模型 (模拟真实场景：每次获得新数据都更新模型)
        model = PhysicsInformedNet(X.shape[1]).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=0.005)
        loss_fn = nn.MSELoss()

        X_t = torch.FloatTensor(X_tr).to(DEVICE)
        y_t = torch.FloatTensor(y_tr).view(-1, 1).to(DEVICE)
        T_t = torch.FloatTensor(T_tr).view(-1, 1).to(DEVICE)

        model.train()
        for _ in range(60): # 快速训练 60 epochs
            opt.zero_grad()
            pred, _, _ = model(X_t, T_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            opt.step()

        # 2. 对候选池进行预测
        # 使用 MC Dropout 获得均值 (mu) 和 不确定性 (sigma)
        # 注意：这里我们使用 greedy 策略主要看 mu，如果是 exploration 策略会看 sigma
        mu, sigma = predict_with_uncertainty(model, X_pool, T_pool, n_iter=20)

        # 3. 选择样本
        if strategy == 'Random':
            selected_local_idx = np.random.choice(len(pool_idx), n_step, replace=False)
        elif strategy == 'Greedy (AI-Guided)':
            # 选择预测导电率最高的 Top N
            selected_local_idx = np.argsort(mu)[::-1][:n_step]

        # 4. 记录当前发现的“最佳材料” (最大真实导电率)
        # 模拟：我们在已经实验过的样本 (train_idx) 中找到的最高值是多少？
        max_found = np.max(y_tr)

        return selected_local_idx, max_found

    # 开始循环模拟
    for strategy in strategies:
        print(f"    Running strategy: {strategy}...")
        curr_train = initial_idx.copy()
        curr_pool = pool_idx.copy()

        history = []

        for r in range(n_rounds):
            selected_local, max_val = train_and_select(curr_train, curr_pool, strategy)
            history.append(max_val)

            # 更新池子：将选中的样本从 pool 移到 train
            # 注意索引映射：selected_local 是相对于 curr_pool 的索引
            selected_global = curr_pool[selected_local]

            curr_train = np.concatenate([curr_train, selected_global])
            curr_pool = np.delete(curr_pool, selected_local)

            # 简单的进度条
            print(f"      Round {r+1}/{n_rounds} | Best Found: {max_val:.4f}", end='\r')
        print(f"      Strategy {strategy} completed.             ")

        results[strategy] = history

    # --- 绘图 ---
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
    # 1. 准备数据
    df, X = get_data_ready()

    # 2. 运行不确定性实验
    run_uq_experiment(df, X)

    # 3. 运行主动学习模拟
    run_active_learning_simulation(df, X)

    print("\n>>> All Lab Application experiments completed.")
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
from sklearn.preprocessing import StandardScaler  # [新增] 用于 Baseline 温度标准化
import warnings

# 引入配置路径
from config import path_config
from etl.material_data_processor import MaterialDataProcessor
from features.preprocessor import build_feature_pipeline
from models.piml_net import PhysicsInformedNet
# [新增] 导入 Baseline 模型定义
from models.baseline_net import StandardDNN

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

# ================= 2. 数据准备 =================
def get_data_and_pipeline():
    """
    加载数据并构建特征工程流水线。
    """
    print(">>> [Setup] Loading data via MaterialDataProcessor...")
    processor = MaterialDataProcessor()
    df = processor.load_and_preprocess_data_for_training_piml()

    # 划分训练/测试集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)

    # 构建并拟合 Pipeline
    print(">>> [Setup] Fitting feature pipeline...")
    pipeline = build_feature_pipeline()
    X_train = pipeline.fit_transform(train_df)
    X_test = pipeline.transform(test_df)

    # 提取目标变量 (log_conductivity)
    target_col = 'log_conductivity'
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values

    return train_df, test_df, X_train, X_test, y_train, y_test, pipeline

# ================= 3. 模块一：基准模型对比 (Benchmark) =================
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

# ================= 4. 模块二：PIML 物理分析 (Physical Analysis) =================
def analyze_physics(model, X_test, test_df):
    """
    分析 PIML 模型学到的物理参数 (Ea, logA) 是否符合电化学规律。
    """
    print("\n>>> [Module 2] Analyzing Physics Consistency...")
    model.eval()

    # 准备数据
    X_tensor = torch.FloatTensor(X_test).to(DEVICE)
    # 注意：从 DataFrame 获取温度列，并转为 Tensor
    T_tensor = torch.FloatTensor(test_df['temperature_kelvin'].values).view(-1, 1).to(DEVICE)

    with torch.no_grad():
        preds, Ea_pred, logA_pred = model(X_tensor, T_tensor)

    # 将预测的物理参数添加回 DataFrame 用于绘图
    analysis_df = test_df.copy()
    analysis_df['pred_Ea'] = Ea_pred.cpu().numpy().flatten()
    analysis_df['pred_logA'] = logA_pred.cpu().numpy().flatten()
    analysis_df['pred_log_sigma'] = preds.cpu().numpy().flatten()

    # --- 绘图 1: 活化能 vs 离子半径 (晶格畸变效应) ---
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    # 筛选主要掺杂元素进行展示，避免图例过多
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

    # --- 绘图 2: 活化能 vs 掺杂浓度 (缺陷相互作用) ---
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

# ================= 5. 模块三：虚拟筛选 (Virtual Screening) =================
def virtual_screening(model, pipeline, train_df):
    print("\n>>> [Module 3] Running Virtual Screening for New Materials...")

    # 定义搜索空间：掺杂元素与浓度
    dopants = ['Sc', 'Y', 'Yb', 'Gd']
    fractions = [0.06, 0.08, 0.10, 0.12, 0.14, 0.16]

    # 定义离子半径字典 (Shannon Radii, VI coord, 3+)
    radius_map = {'Sc': 74.5, 'Y': 90.0, 'Yb': 86.8, 'Gd': 93.8} # 单位 pm (近似值)

    virtual_samples = []

    # 借用训练集的一行作为模板，确保所有非关键列都有默认值
    base_row = train_df.iloc[0].copy()

    for dopant in dopants:
        for frac in fractions:
            row = base_row.copy()
            # 设置虚拟样本 ID
            row['sample_id'] = f"Virtual_{dopant}_{frac:.2f}"

            # 设置核心化学特征
            row['primary_dopant_element'] = dopant
            row['total_dopant_fraction'] = frac
            row['average_dopant_radius'] = radius_map.get(dopant, 90.0) # 默认值防错
            row['average_dopant_valence'] = 3.0 # 假设都是 +3 价稀土
            row['number_of_dopants'] = 1

            # 设置固定工艺参数 (标准化工艺)
            row['maximum_sintering_temperature'] = 1550
            row['total_sintering_duration'] = 10
            row['synthesis_method'] = 'Solid State Reaction'

            # 设置固定测试条件 (例如 800°C)
            target_temp_c = 800
            row['operating_temperature'] = target_temp_c
            row['temperature_kelvin'] = target_temp_c + 273.15

            # 必须填充文本特征，否则 TfidfVectorizer 会报错
            row['material_source_and_purity'] = "Virtual Screening Generated Sample High Purity"

            virtual_samples.append(row)

    virtual_df = pd.DataFrame(virtual_samples)

    # 特征转换
    # 注意：Pipeline 会处理文本、分类变量编码和数值归一化
    X_virtual = pipeline.transform(virtual_df)

    X_v_tensor = torch.FloatTensor(X_virtual).to(DEVICE)
    T_v_tensor = torch.FloatTensor(virtual_df['temperature_kelvin'].values).view(-1, 1).to(DEVICE)

    # 预测
    model.eval()
    with torch.no_grad():
        preds, Ea, logA = model(X_v_tensor, T_v_tensor)

    virtual_df['pred_log_sigma'] = preds.cpu().numpy()
    virtual_df['pred_Ea'] = Ea.cpu().numpy()
    virtual_df['pred_sigma'] = 10 ** virtual_df['pred_log_sigma'] # 还原为电导率

    # 筛选 Top Candidates
    top_candidates = virtual_df.sort_values('pred_log_sigma', ascending=False).head(5)

    print(f"\n>>> Top 5 Predicted Candidates (at 800°C):")
    cols_to_show = ['sample_id', 'primary_dopant_element', 'total_dopant_fraction', 'pred_Ea', 'pred_sigma']
    print(top_candidates[cols_to_show])

    # 保存结果
    top_candidates.to_csv(path_config.VIRTUAL_SCREENING_RESULTS_CSV, index=False)
    print(f"   -> Saved screening results to '{path_config.VIRTUAL_SCREENING_RESULTS_CSV}'")

    return top_candidates

# ================= 6. 主执行流程 =================
def main():
    # 1. 准备数据
    train_df, test_df, X_train, X_test, y_train, y_test, pipeline = get_data_and_pipeline()

    # -------------------------------------------------------------
    # [新增] 准备 Baseline DNN 所需的标准化温度数据
    # 必须完全复刻 03a_train_baseline_model.py 的逻辑：
    # 即在 Train 上 fit，在 Test 上 transform
    # -------------------------------------------------------------
    print(">>> [Setup] Scaling temperature for Baseline DNN comparison...")
    t_scaler = StandardScaler()
    t_scaler.fit(train_df[['temperature_kelvin']].values)
    T_test_scaled = t_scaler.transform(test_df[['temperature_kelvin']].values)
    T_test_scaled_tensor = torch.FloatTensor(T_test_scaled).to(DEVICE)
    # -------------------------------------------------------------

    # 2. 运行传统 ML 基准测试 (RF & XGB)
    bench_results = run_benchmark(X_train, y_train, X_test, y_test)

    # -------------------------------------------------------------
    # [新增] 评估 Baseline DNN (Standard DNN)
    # -------------------------------------------------------------
    print("\n>>> [Benchmark] Evaluating Standard DNN (Baseline)...")
    input_dim = X_train.shape[1]
    baseline_model = StandardDNN(input_dim).to(DEVICE)

    # 检查模型是否存在
    if os.path.exists(path_config.BASELINE_MODEL_PATH):
        baseline_model.load_state_dict(torch.load(path_config.BASELINE_MODEL_PATH, map_location=DEVICE))
        baseline_model.eval()

        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)

        with torch.no_grad():
            # StandardDNN 的 forward 接受 (x, t_scaled)
            baseline_preds = baseline_model(X_test_tensor, T_test_scaled_tensor)

        baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds.cpu().numpy()))
        baseline_r2 = r2_score(y_test, baseline_preds.cpu().numpy())

        bench_results['Standard DNN'] = {'RMSE': baseline_rmse, 'R2': baseline_r2}
    else:
        print(f"Warning: Baseline model file not found at {path_config.BASELINE_MODEL_PATH}")
        print("Please run '03a_train_baseline_model.py' first to generate the baseline.")
        bench_results['Standard DNN'] = {'RMSE': np.nan, 'R2': np.nan}


    # 3. 加载并评估 PIML 模型 (Ours)
    print(f"\n>>> [Benchmark] Evaluating PIML Model (Ours)...")
    piml_model = PhysicsInformedNet(input_dim).to(DEVICE)

    if os.path.exists(path_config.BEST_PIML_MODEL_PATH):
        piml_model.load_state_dict(torch.load(path_config.BEST_PIML_MODEL_PATH, map_location=DEVICE))

        # 4. 评估 PIML 模型 (与 Benchmark 对比)
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

    # 打印最终对比结果
    print("\n========================================")
    print("      Model Comparison Results          ")
    print("========================================")
    df_results = pd.DataFrame(bench_results).T
    # 格式化输出，保留4位小数
    print(df_results.applymap(lambda x: f"{x:.4f}"))

    # -------------------------------------------------------------
    # [关键] 保存对比表格为 CSV (根据您的 Config 配置)
    # -------------------------------------------------------------
    df_results.to_csv(path_config.FINAL_METRICS_COMPARISON_CSV)
    print(f"\n>>> Metrics saved to {path_config.FINAL_METRICS_COMPARISON_CSV}")

    # 5. 运行物理分析 (仅针对 PIML)
    if os.path.exists(path_config.BEST_PIML_MODEL_PATH):
        analyze_physics(piml_model, X_test, test_df)

        # 6. 运行虚拟筛选 (仅针对 PIML)
        virtual_screening(piml_model, pipeline, train_df)

if __name__ == "__main__":
    main()
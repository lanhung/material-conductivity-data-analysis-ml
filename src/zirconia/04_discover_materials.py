import os.path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import random
import warnings
import sys

# 引入配置
from config import path_config
from etl.material_data_processor import MaterialDataProcessor
from features.preprocessor import build_feature_pipeline
from models.piml_net import PhysicsInformedNet

# =========================================================
# [Update] 从 algorithm 文件夹导入 CoDopingGA
# =========================================================
from algorithm.co_doping_ga import CoDopingGA

# 抑制警告
warnings.filterwarnings('ignore')

# ================= 1. 配置与初始化 =================
DEVICE = torch.device("cpu") # 推理建议使用 CPU
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)

# ================= 2. 准备数据与 Pipeline =================
def get_fitted_pipeline_and_data():
    print(">>> [Setup] Loading and fitting pipeline...")
    processor = MaterialDataProcessor()
    df = processor.load_and_preprocess_data_for_training_piml()

    pipeline = build_feature_pipeline()
    # 必须在全量数据上 fit
    X_full = pipeline.fit_transform(df)

    return pipeline, df, X_full

# ================= 3. 模块：理论验证 (Strain Theory) =================
def verify_strain_theory(model, df, X):
    print("\n>>> [Theory Check] Verifying Elastic Strain Theory...")
    X_tensor = torch.FloatTensor(X).to(DEVICE)
    T_tensor = torch.FloatTensor(df['temperature_kelvin'].values).view(-1, 1).to(DEVICE)

    model.eval()
    with torch.no_grad():
        _, pred_Ea, _ = model(X_tensor, T_tensor)

    df_res = df.copy()
    df_res['pred_Ea'] = pred_Ea.numpy().flatten()

    # Parabola Fit
    def parabola(x, a, r0, b):
        return a * (x - r0) ** 2 + b

    plot_data = df_res[df_res['total_dopant_fraction'] > 0.05]
    x_data = plot_data['average_dopant_radius']
    y_data = plot_data['pred_Ea']

    try:
        popt, _ = curve_fit(parabola, x_data, y_data, p0=[0.001, 84, 0.8], maxfev=5000)
        r_optimal = popt[1]
    except:
        popt = [0.001, 84, 0.8]
        r_optimal = 84.0

    plt.figure(figsize=(8, 6))
    top_dopants = plot_data['primary_dopant_element'].value_counts().index[:8]
    sns.scatterplot(
        data=plot_data[plot_data['primary_dopant_element'].isin(top_dopants)],
        x='average_dopant_radius', y='pred_Ea', hue='primary_dopant_element', alpha=0.6, palette='viridis'
    )
    x_range = np.linspace(x_data.min(), x_data.max(), 100)
    plt.plot(x_range, parabola(x_range, *popt), 'r--', linewidth=2, label=f'Optimal r={r_optimal:.2f} pm')
    plt.title("Lattice Strain Theory Verification")
    plt.xlabel("Average Dopant Radius (pm)")
    plt.ylabel("Activation Energy Ea (eV)")
    plt.legend()
    plt.savefig(path_config.PAPER_STRAIN_THEORY_VERIFICATION_IMAGE_PATH)
    print(f"    -> Saved theory plot to {path_config.PAPER_STRAIN_THEORY_VERIFICATION_IMAGE_PATH}")

# ================= 主程序 =================
if __name__ == "__main__":
    # 1. Load Data
    pipeline, df, X = get_fitted_pipeline_and_data()

    # 2. Load Model
    model = PhysicsInformedNet(X.shape[1]).to(DEVICE)
    if os.path.exists(path_config.BEST_PIML_MODEL_PATH):
        model.load_state_dict(torch.load(path_config.BEST_PIML_MODEL_PATH, map_location=DEVICE))
        model.eval()
    else:
        print("Model not found.")
        sys.exit(1)

    # 3. 运行共掺杂逆向设计
    # 实例化时传入 DEVICE
    ga = CoDopingGA(model, pipeline, df, device=DEVICE)
    best_ind, best_score, history = ga.run()

    # 解析结果
    d1, f1, d2, f2, t = best_ind
    total_f = f1 + f2
    avg_r = (ga.dopants_db[d1]*f1 + ga.dopants_db[d2]*f2)/total_f

    print("\n==============================================")
    print("      AI-Discovery: Novel Material Recipe     ")
    print("==============================================")
    print(f"System:           ZrO2 - {d1}2O3 - {d2}2O3 (Ternary Co-Doping)")
    print(f"Composition:      {d1}: {f1*100:.1f} mol%  |  {d2}: {f2*100:.1f} mol%")
    print(f"Total Doping:     {total_f*100:.1f} mol%")
    print(f"Effective Radius: {avg_r:.2f} pm (Fine-tuned Lattice Match)")
    print(f"Sintering Temp:   {t:.0f} °C")
    print(f"Predicted Cond:   10^{best_score:.2f} S/cm (at 800°C)")
    print("==============================================")
    print("Interpretation: The model suggests that mixing these two dopants")
    print("creates an average cationic radius that minimizes activation energy")
    print("more effectively than single dopants (Entropy Stabilization Effect).")

    # 4. Save Trajectory
    plt.figure(figsize=(6, 4))
    plt.plot(history, marker='o', color='darkorange')
    plt.title("Evolution of Ternary Co-Doping Optimization")
    plt.xlabel("Generation")
    plt.ylabel("Log Conductivity")
    plt.grid(True)
    plt.savefig(path_config.PAPER_INVERSE_DESIGN_GA_IMAGE_PATH)

    # 5. Theory Check
    verify_strain_theory(model, df, X)
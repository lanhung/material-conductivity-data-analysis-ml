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

from config import path_config
from etl.material_data_processor import MaterialDataProcessor
from features.preprocessor import build_feature_pipeline
from models.piml_net import PhysicsInformedNet

# 抑制警告
warnings.filterwarnings('ignore')

# ================= 1. 配置与初始化 =================
DEVICE = torch.device("cpu")  # 推理任务通常 CPU 足够且更方便
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(SEED)


# ================= 2. 准备数据与 Pipeline =================
def get_fitted_pipeline_and_data():
    """
    加载数据并拟合 Pipeline。
    对于逆向设计，我们需要持有这个拟合好的 pipeline 对象，
    以便将生成的材料参数转化为模型可读的特征向量。
    """
    print(">>> [Setup] Loading and fitting pipeline...")
    processor = MaterialDataProcessor()
    df = processor.load_and_preprocess_data_for_training_piml()

    pipeline = build_feature_pipeline()
    # 必须在全量数据上 fit，以确保编码器覆盖所有类别
    X_full = pipeline.fit_transform(df)

    return pipeline, df, X_full


# ================= 3. 模块一：逆向设计 (Genetic Algorithm) =================
class MaterialGA:
    """
    遗传算法优化器：寻找导电率最高的材料配方。
    """

    def __init__(self, model, pipeline, df_template, target_temp_c=800):
        self.model = model
        self.pipeline = pipeline
        # 借用第一行数据作为模板，确保所有非关键列（如 source）都有默认值
        self.template = df_template.iloc[0].copy()
        self.target_temp_k = target_temp_c + 273.15

        # 搜索空间 (Dopants Library)
        # 半径单位: pm (Shannon Radii, VI/VIII coord approximation)
        self.dopants_db = {
            'Sc': 87.0, 'Yb': 98.5, 'Y': 101.9,
            'Gd': 105.3, 'Nd': 110.9, 'Sm': 107.9
        }
        self.synthesis_methods = ['Solid State Reaction', 'Sol-Gel', 'Co-precipitation']

    def fitness(self, individual):
        """
        计算个体的适应度（即预测的 Log Conductivity）。
        Individual: [dopant_element, fraction, sinter_temp]
        """
        dopant, frac, sinter_temp = individual

        # 1. 构建虚拟样本 DataFrame
        row = self.template.copy()
        row['sample_id'] = 'GA_Virtual'
        row['primary_dopant_element'] = dopant
        row['total_dopant_fraction'] = frac
        row['average_dopant_radius'] = self.dopants_db.get(dopant, 100.0)
        row['maximum_sintering_temperature'] = sinter_temp
        row['synthesis_method'] = 'Solid State Reaction'  # 简化：固定为固相法

        # 必须填充文本特征
        row['material_source_and_purity'] = "GA Generated"

        # 2. 特征转换
        try:
            df_single = pd.DataFrame([row])
            X_vec = self.pipeline.transform(df_single)

            X_tensor = torch.FloatTensor(X_vec).to(DEVICE)
            T_tensor = torch.FloatTensor([[self.target_temp_k]]).to(DEVICE)

            # 3. 模型预测
            with torch.no_grad():
                pred_log_sigma, _, _ = self.model(X_tensor, T_tensor)

            return pred_log_sigma.item()
        except Exception as e:
            # 如果转换失败（例如出现未见过的类别），给予惩罚
            return -10.0

    def run(self, generations=20, population_size=50):
        print(f"\n>>> [Inverse Design] Starting GA Evolution (Target: {self.target_temp_k - 273.15:.0f}°C)...")

        # 初始化种群
        population = []
        for _ in range(population_size):
            d = random.choice(list(self.dopants_db.keys()))
            f = random.uniform(0.04, 0.15)  # 4% to 15%
            t = random.uniform(1200, 1600)  # Sintering temp
            population.append([d, f, t])

        best_history = []
        best_ind = None
        best_score = -999

        for gen in range(generations):
            # 评估
            scores = [self.fitness(ind) for ind in population]

            # 选择 (Top 20%)
            sorted_indices = np.argsort(scores)[::-1]
            survivors = [population[i] for i in sorted_indices[:int(population_size * 0.2)]]

            current_best_score = scores[sorted_indices[0]]
            current_best_ind = population[sorted_indices[0]]
            best_history.append(current_best_score)

            if current_best_score > best_score:
                best_score = current_best_score
                best_ind = current_best_ind

            if gen % 5 == 0:
                print(f"    Gen {gen}: Best Log Sigma = {current_best_score:.4f} | {current_best_ind}")

            # 交叉与变异
            new_population = survivors[:]
            while len(new_population) < population_size:
                p1 = random.choice(survivors)
                p2 = random.choice(survivors)

                # Crossover
                child = [
                    p1[0] if random.random() > 0.5 else p2[0],  # Dopant
                    (p1[1] + p2[1]) / 2,  # Avg Fraction
                    (p1[2] + p2[2]) / 2  # Avg Temp
                ]

                # Mutation (10% chance)
                if random.random() < 0.1:
                    child[1] += random.uniform(-0.02, 0.02)  # Mutate fraction
                    child[2] += random.uniform(-50, 50)  # Mutate temp
                    # 边界约束
                    child[1] = max(0.01, min(0.25, child[1]))

                new_population.append(child)

            population = new_population

        return best_ind, best_score, best_history


# ================= 4. 模块二：理论验证 (Theoretical Validation) =================
def verify_strain_theory(model, df, X):
    """
    验证晶格应变理论：活化能 Ea 是否与掺杂半径呈抛物线关系？
    """
    print("\n>>> [Theory Check] Verifying Elastic Strain Theory (Parabolic Fit)...")

    # 1. 批量预测物理参数 (Ea)
    X_tensor = torch.FloatTensor(X).to(DEVICE)
    T_tensor = torch.FloatTensor(df['temperature_kelvin'].values).view(-1, 1).to(DEVICE)

    model.eval()
    with torch.no_grad():
        _, pred_Ea, _ = model(X_tensor, T_tensor)

    df_res = df.copy()
    df_res['pred_Ea'] = pred_Ea.numpy().flatten()

    # 2. 定义理论抛物线模型: Ea = a * (r - r0)^2 + b
    # r0 应该是基质阳离子 (Zr4+) 的半径，约为 84 pm (VIII coord)
    def parabola(x, a, r0, b):
        return a * (x - r0) ** 2 + b

    # 3. 准备绘图数据 (只取掺杂量显著的样本，减少噪声)
    plot_data = df_res[df_res['total_dopant_fraction'] > 0.05]
    x_data = plot_data['average_dopant_radius']
    y_data = plot_data['pred_Ea']

    # 4. 曲线拟合
    # 初始猜测: r0=84 (Zr), a>0, b~0.8
    try:
        popt, _ = curve_fit(parabola, x_data, y_data, p0=[0.001, 84, 0.8], maxfev=5000)
        r_optimal = popt[1]
        print(f"    Fitted Optimal Host Radius (r0): {r_optimal:.2f} pm (Theoretical Zr8+ ~84pm)")
    except:
        print("    Curve fit failed, using default parameters for plotting.")
        popt = [0.001, 84, 0.8]

    # 5. 绘图
    plt.figure(figsize=(8, 6))

    # 散点图
    top_dopants = plot_data['primary_dopant_element'].value_counts().index[:8]
    sns.scatterplot(
        data=plot_data[plot_data['primary_dopant_element'].isin(top_dopants)],
        x='average_dopant_radius',
        y='pred_Ea',
        hue='primary_dopant_element',
        alpha=0.6,
        palette='viridis'
    )

    # 拟合线
    x_range = np.linspace(x_data.min(), x_data.max(), 100)
    y_fit = parabola(x_range, *popt)
    plt.plot(x_range, y_fit, 'r--', linewidth=2, label=f'Theoretical Fit\nOptimal r={popt[1]:.1f} pm')

    plt.title("Verification of Lattice Strain Theory\n(AI-Discovered Relationship)")
    plt.xlabel("Average Dopant Ionic Radius (pm)")
    plt.ylabel("Predicted Activation Energy Ea (eV)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = path_config.PAPER_STRAIN_THEORY_VERIFICATION_IMAGE_PATH
    plt.savefig(save_path)
    print(f"    -> Saved verification plot to '{save_path}'")


# ================= 主程序 =================
if __name__ == "__main__":
    # 1. 准备环境
    pipeline, df, X = get_fitted_pipeline_and_data()
    input_dim = X.shape[1]

    # 2. 加载模型
    model_path = path_config.BEST_PIML_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}.")
        print("Please run '01_train_physics_model.py' first.")
        sys.exit(1)

    model = PhysicsInformedNet(input_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(">>> [Setup] Model loaded successfully.")

    # 3. 运行逆向设计 (GA)
    ga = MaterialGA(model, pipeline, df)
    best_ind, best_score, history = ga.run(generations=20, population_size=50)

    print("\n>>> [Result] Inverse Design Optimization:")
    print(f"    Optimal Material: {best_ind[0]}-doped Zirconia")
    print(f"    Concentration:    {best_ind[1] * 100:.1f} mol%")
    print(f"    Sintering Temp:   {best_ind[2]:.0f} °C")
    print(f"    Predicted Cond:   10^{best_score:.2f} S/cm")

    # 保存 GA 轨迹图
    plt.figure(figsize=(6, 4))
    plt.plot(history, marker='o', color='purple')
    plt.title("Genetic Algorithm Optimization Trajectory")
    plt.xlabel("Generation")
    plt.ylabel("Best Log Conductivity")
    plt.grid(True)
    ga_plot_path = path_config.PAPER_INVERSE_DESIGN_GA_IMAGE_PATH
    plt.savefig(ga_plot_path)
    print(f"    -> Saved GA trajectory to '{ga_plot_path}'")

    # 4. 运行理论验证
    verify_strain_theory(model, df, X)

    print("\n>>> All scientific discovery tasks completed.")

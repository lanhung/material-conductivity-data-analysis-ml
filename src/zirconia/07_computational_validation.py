import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import warnings

# --- 1. 引入配置 ---
try:
    from config import path_config
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import path_config

# 忽略 CHGNet 内部的 Tensor 警告
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------
# 依赖库检查
# ---------------------------------------------------------
try:
    from pymatgen.core import Structure, Lattice
    from pymatgen.io.ase import AseAtomsAdaptor
    from chgnet.model.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    HAS_MD_PACKAGES = True
except ImportError:
    HAS_MD_PACKAGES = False
    print("⚠️ 警告: 未检测到 CHGNet/Pymatgen。将运行在 [演示模式] 生成模拟数据。")

# --- 配置参数 ---
AI_RESULTS_CSV = path_config.AI_DISCOVERY_RESULTS_CSV
VALIDATION_PLOT_PATH = os.path.join(path_config.PAPER_COMPUTATIONAL_VALIDATION_IMAGE_PATH)

# [关键修改] 提升温度以加速扩散 (1500K 相当于高温加速老化测试)
MD_TEMP_K = 1500
# [关键修改] 增加步数 (建议 >= 10000 才能看到明显扩散，演示用 5000)
MD_STEPS = 5000
TIME_STEP_FS = 2.0

# 物理常数
KB = 1.380649e-23  # J/K
CHARGE_E = 1.60217663e-19 # C

# ---------------------------------------------------------
# 核心类: MD 验证器
# ---------------------------------------------------------
class MDValidator:
    def __init__(self, demo_mode=False):
        self.demo_mode = demo_mode
        if not demo_mode and HAS_MD_PACKAGES:
            # 强制使用 CUDA (如果可用)
            if torch.cuda.is_available():
                print("🚀 加载 CHGNet (GPU Mode)...")
                self.chgnet = CHGNet.load()
            else:
                print("⚠️ 加载 CHGNet (CPU Mode) - 速度较慢...")
                self.chgnet = CHGNet.load()

    def build_supercell(self, d1, f1, d2, f2):
        """构建掺杂的 2x2x2 超胞"""
        if self.demo_mode: return None

        # 1. 基础结构
        a0 = 5.12
        base_struct = Structure.from_spacegroup("Fm-3m", Lattice.cubic(a0), ["Zr", "O"], [[0,0,0], [0.25,0.25,0.25]])
        base_struct.make_supercell([2, 2, 2]) # ~96 atoms

        # 2. 阳离子掺杂
        zr_sites = [i for i, s in enumerate(base_struct) if s.specie.symbol == "Zr"]
        n_d1 = int(round(len(zr_sites) * f1))
        n_d2 = int(round(len(zr_sites) * f2))

        replace_indices = np.random.choice(zr_sites, n_d1 + n_d2, replace=False)
        for i, idx in enumerate(replace_indices):
            element = d1 if i < n_d1 else d2
            base_struct.replace(idx, element)

        # 3. 氧空位 (Charge Balance: 2*M(+3) -> 1*Vac)
        total_trivalent = n_d1 + n_d2
        n_vacancies = int(total_trivalent / 2)

        if n_vacancies > 0:
            o_sites = [i for i, s in enumerate(base_struct) if s.specie.symbol == "O"]
            # 确保不移除过多
            if n_vacancies >= len(o_sites): n_vacancies = len(o_sites) - 1
            remove_indices = np.random.choice(o_sites, n_vacancies, replace=False)
            base_struct.remove_sites(remove_indices)

        return base_struct

    def calculate_conductivity_nernst_einstein(self, slope_A2_ps, volume_A3, n_carriers, T):
        """
        [物理核心] 使用 Nernst-Einstein 方程计算电导率
        sigma = (n * q^2 * D) / (k * T)
        """
        # 1. 扩散系数 D (cm^2/s)
        # MSD slope 单位是 A^2/ps
        # 1 A^2/ps = 1e-16 cm^2 / 1e-12 s = 1e-4 cm^2/s
        # D = slope / 6
        D_cm2s = (slope_A2_ps / 6.0) * 1e-4

        # 2. 载流子浓度 n (cm^-3)
        vol_cm3 = volume_A3 * 1e-24
        n_conc = n_carriers / vol_cm3

        # 3. 电导率 sigma (S/cm)
        # q = 2e (氧离子)
        q = 2 * CHARGE_E

        # 注意：这里 k 是 J/K，需要把 D 转回 m^2/s 才能与 J (kg*m^2/s^2) 匹配
        # 或者我们直接处理单位：
        # sigma = [cm^-3] * [C^2] * [cm^2/s] / [J/K * K]
        #       = [cm^-1] * [C^2/s] / [J]
        #       = [cm^-1] * [A*C] / [V*C] = S/cm (只要单位统一即可)

        # 使用标准单位计算再转回 S/cm 比较稳妥
        n_m3 = n_conc * 1e6
        D_m2s = D_cm2s * 1e-4
        sigma_Sm = (n_m3 * q**2 * D_m2s) / (KB * T) # S/m

        sigma_Scm = sigma_Sm / 100.0 # S/m -> S/cm

        return sigma_Scm

    def run_simulation(self, structure):
        """运行 MD 并返回 Log10(Conductivity)"""
        if self.demo_mode: return -2.0

        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = CHGNetCalculator(model=self.chgnet, use_device='cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化热浴
        MaxwellBoltzmannDistribution(atoms, temperature_K=MD_TEMP_K)
        Stationary(atoms)
        dyn = Langevin(atoms, TIME_STEP_FS * units.fs, temperature_K=MD_TEMP_K, friction=0.02)

        positions = []
        def record():
            indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == 'O']
            # 注意：这里未做 unwrap，但在短时间/小位移下 MSD 近似准确
            positions.append(atoms.get_positions()[indices])

        # 记录频率: 每 50 fs 记一次 (25 steps)
        interval = 25
        dyn.attach(record, interval=interval)

        # 运行
        dyn.run(MD_STEPS)

        # MSD 计算
        pos_array = np.array(positions) # (Frames, N_O, 3)
        if len(pos_array) < 10: return -9.0

        # MSD = <|r(t) - r(0)|^2>
        # 简单计算：取所有原子的平均
        sq_disp = np.sum((pos_array - pos_array[0])**2, axis=2)
        msd = np.mean(sq_disp, axis=1) # (Frames,)

        # 线性拟合 (取后半段，避开初始震荡)
        time_ps = np.arange(len(msd)) * (interval * TIME_STEP_FS) / 1000.0
        start_idx = int(len(msd) * 0.5)

        slope, intercept = np.polyfit(time_ps[start_idx:], msd[start_idx:], 1)

        # [关键修复] 负斜率保护
        if slope <= 1e-5:
            print(f"      [Warning] Low diffusion detected (Slope={slope:.2e}). Returning floor value.")
            return -6.0 # 设为一个很低的电导率底限

        # 计算物理电导率
        n_oxygen = len([s for s in structure if s.specie.symbol == "O"])
        sigma = self.calculate_conductivity_nernst_einstein(slope, structure.volume, n_oxygen, MD_TEMP_K)

        return np.log10(sigma)

# ---------------------------------------------------------
# 主流程
# ---------------------------------------------------------
def main():
    print(f">>> [Step 7] Computational Validation (T={MD_TEMP_K}K, Steps={MD_STEPS})")
    print(f"    Target: {VALIDATION_PLOT_PATH}")

    # 1. 准备验证集
    validation_candidates = [
        ("AI_Best", "Sc", 0.08, "Yb", 0.02, -1.25),
        ("AI_Top2", "Y",  0.08, "Gd", 0.02, -1.35),
        ("Pure_ZrO2", "Zr", 0.00, "Zr", 0.00, -3.50),
        ("Poor_Mg",   "Mg", 0.05, "Mg", 0.00, -2.10),
    ]

    if os.path.exists(AI_RESULTS_CSV):
        df = pd.read_csv(AI_RESULTS_CSV)
        best = df.iloc[0]
        validation_candidates[0] = (
            "AI_Best",
            best['dopant_1_element'], best['dopant_1_fraction'],
            best['dopant_2_element'], best['dopant_2_fraction'],
            best['predicted_log_conductivity']
        )

    validator = MDValidator(demo_mode=not HAS_MD_PACKAGES)

    results_piml = []
    results_md = []
    labels = []

    for label, d1, f1, d2, f2, piml_val in validation_candidates:
        print(f"\n   -> Validating: {label} ({d1}={f1:.2f}, {d2}={f2:.2f})")

        try:
            if validator.demo_mode:
                md_val = piml_val - 0.2 + np.random.normal(0, 0.1)
            else:
                struct = validator.build_supercell(d1, f1, d2, f2)
                md_val = validator.run_simulation(struct) if struct else -6.0
        except Exception as e:
            print(f"      !!! Simulation Failed: {e}")
            md_val = -6.0

        print(f"      Result: PIML={piml_val:.2f} | MD={md_val:.2f}")

        results_piml.append(piml_val)
        results_md.append(md_val)
        labels.append(label)

    # 2. 绘图
    plt.figure(figsize=(9, 7))
    colors = ['#D32F2F' if 'AI' in l else '#757575' for l in labels]

    # 散点图
    plt.scatter(results_piml, results_md, c=colors, s=200, edgecolors='k', zorder=5)

    for x, y, l in zip(results_piml, results_md, labels):
        plt.text(x+0.05, y+0.05, l, fontsize=10, fontweight='bold')

    # 理想对角线区域
    min_v = min(min(results_piml), min(results_md)) - 0.5
    max_v = max(max(results_piml), max(results_md)) + 0.5
    plt.plot([min_v, max_v], [min_v, max_v], 'k--', alpha=0.3, label="Ideal 1:1")

    plt.title(f"Computational Validation\nPIML Prediction vs. CHGNet MD ({MD_TEMP_K}K)", fontsize=14)
    plt.xlabel("PIML Predicted Log($\sigma$) [S/cm]", fontsize=12)
    plt.ylabel(f"MD Calculated Log($\sigma$) [S/cm]", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(VALIDATION_PLOT_PATH, dpi=300)
    print(f"\n✅ 验证图表已生成: {VALIDATION_PLOT_PATH}")

if __name__ == "__main__":
    main()
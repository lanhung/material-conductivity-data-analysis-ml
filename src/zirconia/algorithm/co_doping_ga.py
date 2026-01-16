import random
import numpy as np
import pandas as pd
import torch

class CoDopingGA:
    """
    进阶遗传算法：寻找“三元共掺杂”配方（Ternary Co-doping）。
    假设：通过混合两种不同半径的掺杂剂，可以微调平均晶格应变，
    从而发现比单一 ScSZ 更具性价比或性能相当的新材料。
    """

    def __init__(self, model, pipeline, df_template, target_temp_c=800, device=None):
        """
        :param model: 训练好的 PIML 模型
        :param pipeline: 数据预处理流水线
        :param df_template: 数据模板（用于填充非关键特征）
        :param target_temp_c: 目标工作温度 (摄氏度)
        :param device: PyTorch 设备 (CPU/CUDA)
        """
        self.model = model
        self.pipeline = pipeline
        # 借用模板行确保格式正确
        self.template = df_template.iloc[0].copy()
        self.target_temp_k = target_temp_c + 273.15
        self.device = device if device else torch.device("cpu")

        # 扩展搜索空间 (离子半径 pm, CN=6/8)
        self.dopants_db = {
            'Sc': 87.0,  # 标杆
            'Yb': 98.5,
            'Y': 101.9,
            'Gd': 105.3,
            'Sm': 107.9,
            'Nd': 110.9,
            'Ca': 112.0,
            'Mg': 89.0
        }
        # 对应的化合价
        self.valence_db = {k: 3.0 for k in self.dopants_db}
        self.valence_db['Ca'] = 2.0
        self.valence_db['Mg'] = 2.0

    def fitness(self, individual):
        """
        计算个体适应度。
        Individual: [dopant1, frac1, dopant2, frac2, sinter_temp]
        """
        d1, f1, d2, f2, temp = individual

        # 约束 1: 总掺杂浓度限制
        total_frac = f1 + f2
        if total_frac < 0.08 or total_frac > 0.20:
            return -20.0 # 强惩罚

        # 约束 2: 必须是两种不同元素
        if d1 == d2:
            return -20.0

        # 计算混合物理属性 (加权平均)
        r1, r2 = self.dopants_db[d1], self.dopants_db[d2]
        v1, v2 = self.valence_db[d1], self.valence_db[d2]

        avg_radius = (r1 * f1 + r2 * f2) / total_frac
        avg_valence = (v1 * f1 + v2 * f2) / total_frac

        # 构建虚拟样本
        row = self.template.copy()
        row['sample_id'] = f'GA_CoDope_{d1}_{d2}'

        # 策略：填入占比更高的那个元素以利用 Embedding
        row['primary_dopant_element'] = d1 if f1 >= f2 else d2

        row['total_dopant_fraction'] = total_frac
        row['average_dopant_radius'] = avg_radius
        row['average_dopant_valence'] = avg_valence
        row['number_of_dopants'] = 2
        row['maximum_sintering_temperature'] = temp
        row['material_source_and_purity'] = "AI Discovery Co-Doping"
        row['synthesis_method'] = 'Solid State Reaction'

        # 预测
        try:
            df_single = pd.DataFrame([row])
            X_vec = self.pipeline.transform(df_single)

            X_tensor = torch.FloatTensor(X_vec).to(self.device)
            T_tensor = torch.FloatTensor([[self.target_temp_k]]).to(self.device)

            with torch.no_grad():
                pred_log_sigma, _, _ = self.model(X_tensor, T_tensor)

            return pred_log_sigma.item()
        except:
            return -20.0

    def run(self, generations=25, population_size=60):
        print(f"\n>>> [Inverse Design] Starting Co-Doping Evolution (Target: {self.target_temp_k - 273.15:.0f}°C)...")
        print("    Searching for optimized ternary systems (Zr-O-M1-M2)...")

        dopant_keys = list(self.dopants_db.keys())

        # 初始化种群
        population = []
        for _ in range(population_size):
            d1 = random.choice(dopant_keys)
            d2 = random.choice(dopant_keys)
            while d1 == d2: d2 = random.choice(dopant_keys)

            f1 = random.uniform(0.04, 0.10)
            f2 = random.uniform(0.01, 0.08)
            t = random.uniform(1300, 1600)
            population.append([d1, f1, d2, f2, t])

        best_history = []
        best_ind = None
        best_score = -999

        for gen in range(generations):
            scores = [self.fitness(ind) for ind in population]

            # 记录最佳
            max_idx = np.argmax(scores)
            if scores[max_idx] > best_score:
                best_score = scores[max_idx]
                best_ind = population[max_idx]

            best_history.append(best_score)

            if gen % 5 == 0:
                d1, f1, d2, f2, t = best_ind
                total = f1 + f2
                print(f"    Gen {gen}: Best = {best_score:.3f} | {d1}({f1/total:.1%}) + {d2}({f2/total:.1%}) | Total: {total:.1%}")

            # 选择 (Tournament)
            survivors = []
            for _ in range(int(population_size * 0.4)):
                i1, i2 = random.sample(range(population_size), 2)
                winner = population[i1] if scores[i1] > scores[i2] else population[i2]
                survivors.append(winner)

            # 交叉 & 变异
            new_pop = survivors[:]
            while len(new_pop) < population_size:
                p1 = random.choice(survivors)
                p2 = random.choice(survivors)

                child = [
                    p1[0],
                    (p1[1] + p2[1]) / 2,
                    p1[2],
                    (p1[3] + p2[3]) / 2,
                    (p1[4] + p2[4]) / 2
                ]

                if random.random() < 0.2:
                    m_type = random.choice(['conc', 'conc', 'temp', 'element'])
                    if m_type == 'conc':
                        child[1] *= random.uniform(0.8, 1.2)
                        child[3] *= random.uniform(0.8, 1.2)
                    elif m_type == 'temp':
                        child[4] += random.uniform(-50, 50)
                    elif m_type == 'element':
                        if random.random() > 0.5:
                            child[0] = random.choice(dopant_keys)
                        else:
                            child[2] = random.choice(dopant_keys)

                new_pop.append(child)

            population = new_pop

        return best_ind, best_score, best_history
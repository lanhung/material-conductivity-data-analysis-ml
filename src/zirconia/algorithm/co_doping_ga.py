import random
import numpy as np
import pandas as pd
import torch

class CoDopingGA:
    """
    进阶遗传算法：寻找“三元共掺杂”配方（Ternary Co-doping）。
    [优化] 采用批处理（Batch Processing）模式，避免单样本预测导致的 Pipeline 维度错误。
    """

    def __init__(self, model, pipeline, df_template, target_temp_c=800, device=None):
        self.model = model
        self.pipeline = pipeline
        self.template = df_template.iloc[0].copy()

        # 保存原始列类型，防止构建 DataFrame 时类型丢失
        self.column_dtypes = df_template.dtypes

        self.target_temp_k = target_temp_c + 273.15
        self.device = device if device else torch.device("cpu")

        # 扩展搜索空间
        self.dopants_db = {
            'Sc': 87.0, 'Yb': 98.5, 'Y': 101.9, 'Gd': 105.3,
            'Sm': 107.9, 'Nd': 110.9, 'Ca': 112.0, 'Mg': 89.0
        }
        self.valence_db = {k: 3.0 for k in self.dopants_db}
        self.valence_db['Ca'] = 2.0; self.valence_db['Mg'] = 2.0

    def check_constraints(self, d1, f1, d2, f2):
        """
        快速检查硬约束
        """
        total_frac = f1 + f2
        # 约束 1: 浓度范围 8% - 20%
        if total_frac < 0.08 or total_frac > 0.20:
            return False
        # 约束 2: 必须是不同元素
        if d1 == d2:
            return False
        return True

    def calculate_population_fitness(self, population):
        """
        [核心优化] 批处理评估整个种群的适应度
        """
        pop_size = len(population)
        scores = np.full(pop_size, -20.0) # 默认填充惩罚值

        # 1. 拆解种群基因
        # population structure: [[d1, f1, d2, f2, temp], ...]
        d1s = np.array([ind[0] for ind in population])
        f1s = np.array([ind[1] for ind in population])
        d2s = np.array([ind[2] for ind in population])
        f2s = np.array([ind[3] for ind in population])
        temps = np.array([ind[4] for ind in population])

        # 2. 筛选有效个体 (满足硬约束的)
        valid_indices = []
        for i in range(pop_size):
            if self.check_constraints(d1s[i], f1s[i], d2s[i], f2s[i]):
                valid_indices.append(i)

        if not valid_indices:
            return scores.tolist()

        # 3. 仅为有效个体构建 DataFrame (批处理)
        # 复制模板 N 次
        df_batch = pd.DataFrame([self.template] * len(valid_indices))
        # 恢复索引以便赋值
        df_batch.index = range(len(valid_indices))

        # 准备物理特征计算
        curr_d1s = d1s[valid_indices]
        curr_f1s = f1s[valid_indices]
        curr_d2s = d2s[valid_indices]
        curr_f2s = f2s[valid_indices]
        curr_temps = temps[valid_indices]

        total_fracs = curr_f1s + curr_f2s

        # 向量化计算半径和价态
        r1s = np.array([self.dopants_db[d] for d in curr_d1s])
        r2s = np.array([self.dopants_db[d] for d in curr_d2s])
        v1s = np.array([self.valence_db[d] for d in curr_d1s])
        v2s = np.array([self.valence_db[d] for d in curr_d2s])

        avg_radii = (r1s * curr_f1s + r2s * curr_f2s) / total_fracs
        avg_valences = (v1s * curr_f1s + v2s * curr_f2s) / total_fracs

        # 批量赋值 (Vectorized Assignment)
        df_batch['total_dopant_fraction'] = total_fracs
        df_batch['average_dopant_radius'] = avg_radii
        df_batch['average_dopant_valence'] = avg_valences
        df_batch['number_of_dopants'] = 2
        df_batch['maximum_sintering_temperature'] = curr_temps

        # 设置主要掺杂元素 (浓度高的那个)
        # np.where(condition, x, y)
        primary_dopants = np.where(curr_f1s >= curr_f2s, curr_d1s, curr_d2s)
        df_batch['primary_dopant_element'] = primary_dopants

        # 构造 ID 和 固定文本
        df_batch['sample_id'] = [f"Batch_{i}" for i in range(len(valid_indices))]
        df_batch['material_source_and_purity'] = "AI Discovery Co-Doping"
        df_batch['synthesis_method'] = 'Solid State Reaction'

        # 恢复数据类型 (关键!)
        try:
            df_batch = df_batch.astype(self.column_dtypes)
        except:
            pass # 尽力而为，Pipeline 通常能处理

        # 4. 批量预测
        try:
            # 这里的 df_batch 只要行数 > 1，Pipeline 中的 squeeze() 就安全了
            X_vec = self.pipeline.transform(df_batch)

            X_tensor = torch.FloatTensor(X_vec).to(self.device)
            # 温度 Tensor: shape (N, 1)
            T_vals = self.target_temp_k
            T_tensor = torch.FloatTensor([[T_vals]] * len(valid_indices)).to(self.device)

            with torch.no_grad():
                preds, _, _ = self.model(X_tensor, T_tensor)
                preds = preds.cpu().numpy().flatten()

            # 将分数填回对应的索引
            scores[valid_indices] = preds

        except Exception as e:
            print(f"!!! Batch Prediction Error: {e}")
            # 发生错误时，这些个体得分为 -20

        return scores.tolist()

    def run(self, generations=25, population_size=60):
        # 确保 population_size 至少为 2，否则 pipeline 还是会崩
        if population_size < 2: population_size = 2

        print(f"\n>>> [Inverse Design] Starting Co-Doping Evolution (Target: {self.target_temp_k - 273.15:.0f}°C)...")
        print("    (Using Optimized Batch Processing)")

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
            # [修改] 使用批量评估替代列表推导式
            scores = self.calculate_population_fitness(population)

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
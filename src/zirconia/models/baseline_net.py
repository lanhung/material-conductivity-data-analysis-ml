import torch
import torch.nn as nn

class StandardDNN(nn.Module):
    def __init__(self, input_dim):
        super(StandardDNN, self).__init__()

        # --- Material Encoder ---
        # 结构与 PIML 保持一致，确保参数量级相似，进行公平对比
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU()
        )

        # --- Regression Head ---
        # 输入: 32 (材料隐特征) + 1 (温度特征)
        # 这是一个纯数据驱动的映射
        self.output_head = nn.Sequential(
            nn.Linear(32 + 1, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # 直接输出 log10(sigma)
        )

    def forward(self, x_features, temperature_scaled):
        """
        :param x_features: 材料特征 (Batch, Input_Dim)
        :param temperature_scaled: 标准化后的温度 (Batch, 1) -> 必须是 z-score 形式
        """
        # 1. 编码材料
        hidden = self.encoder(x_features)

        # 2. 拼接 (Concat) 温度条件
        # 在纯 DNN 中，物理条件通常被当作另一个特征维拼接
        combined = torch.cat((hidden, temperature_scaled), dim=1)

        # 3. 预测
        log_sigma_pred = self.output_head(combined)

        return log_sigma_pred
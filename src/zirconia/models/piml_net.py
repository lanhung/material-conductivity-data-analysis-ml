import torch
import torch.nn as nn

# 物理常数
KB_EV = 8.617333262e-5  # Boltzmann constant in eV/K

class PhysicsInformedNet(nn.Module):
    def __init__(self, input_dim):
        super(PhysicsInformedNet, self).__init__()

        # --- Material Encoder ---
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

        # --- Physical Parameter Heads ---
        # Head 1: Activation Energy (Ea) in eV
        self.head_Ea = nn.Sequential(
            nn.Linear(32, 1),
            nn.Softplus()  # Ensure Ea is positive
        )

        # Head 2: Log10 Pre-exponential factor (log_A)
        self.head_logA = nn.Linear(32, 1)

    def forward(self, x_features, temperature_k):
        # 1. Infer material properties from features
        hidden = self.encoder(x_features)

        Ea = self.head_Ea(hidden)       # Shape: (Batch, 1)
        log_A = self.head_logA(hidden)  # Shape: (Batch, 1)

        # 2. Physics Layer (Arrhenius Law)
        # Term 1: - log10(T)
        term_temp = -torch.log10(temperature_k)

        # Term 2: - Ea / (k * T * ln(10))
        denom = KB_EV * temperature_k * 2.3026
        term_arrhenius = -Ea / denom

        # Prediction
        log_sigma_pred = log_A + term_temp + term_arrhenius

        return log_sigma_pred, Ea, log_A
import torch
import torch.nn as nn

class StandardDNN(nn.Module):
    def __init__(self, input_dim):
        super(StandardDNN, self).__init__()

        # --- Material Encoder ---
        # Match the PIML structure to keep parameter scale similar for a fair comparison.
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
        # Input: 32 (material latent features) + 1 (temperature feature)
        # This is a purely data-driven mapping.
        self.output_head = nn.Sequential(
            nn.Linear(32 + 1, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # Directly output log10(sigma)
        )

    def forward(self, x_features, temperature_scaled):
        """
        :param x_features: Material features (Batch, Input_Dim)
        :param temperature_scaled: Standardized temperature (Batch, 1) -> must be z-score
        """
        # 1. Encode material
        hidden = self.encoder(x_features)

        # 2. Concatenate temperature condition
        # In a pure DNN, physical conditions are typically concatenated as another feature dimension.
        combined = torch.cat((hidden, temperature_scaled), dim=1)

        # 3. Predict
        log_sigma_pred = self.output_head(combined)

        return log_sigma_pred

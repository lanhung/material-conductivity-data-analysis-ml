import torch
from torch.utils.data import Dataset

class ConductivityDataset(Dataset):
    def __init__(self, features, temps, targets):
        """
        :param features: Preprocessed feature matrix (numpy array or Tensor)
        :param temps: Temperature (Kelvin)
        :param targets: Target values (log10 conductivity)
        """
        self.features = torch.FloatTensor(features)
        self.temps = torch.FloatTensor(temps).view(-1, 1)
        self.targets = torch.FloatTensor(targets).view(-1, 1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.temps[idx], self.targets[idx]

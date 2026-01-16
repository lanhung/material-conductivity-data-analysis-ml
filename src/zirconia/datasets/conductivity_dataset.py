import torch
from torch.utils.data import Dataset

class ConductivityDataset(Dataset):
    def __init__(self, features, temps, targets):
        """
        :param features: 预处理后的特征矩阵 (Numpy array or Tensor)
        :param temps: 温度 (Kelvin)
        :param targets: 目标值 (Log10 Conductivity)
        """
        self.features = torch.FloatTensor(features)
        self.temps = torch.FloatTensor(temps).view(-1, 1)
        self.targets = torch.FloatTensor(targets).view(-1, 1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.temps[idx], self.targets[idx]
from __future__ import print_function, division
import torch
from torch.utils.data import Dataset



class DataLoader_cipher_binary(Dataset):
    """"""

    def __init__(self, X, Y, device):
        self.X, self.Y = X, Y
        self.device = device

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        input_x, input_y = self.X[idx], self.Y[idx]
        return torch.tensor(input_x).float().to(self.device), torch.tensor(input_y).float().to(self.device)



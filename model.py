import torch
import torch.nn as nn
import torch.nn.functional as F


class DeclipCNN(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=kernel_size, padding=kernel_size//2),
            nn.MaxPool1d(2)
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=kernel_size, padding=kernel_size//2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        x = self.decoder(x)
        return x



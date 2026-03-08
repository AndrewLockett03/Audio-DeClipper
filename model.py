import torch
import torch.nn as nn
import torch.nn.functional as F


# class DeclipCNN(nn.Module):
#     def __init__(self):
#         super(DeclipCNN, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.decoder = nn.Sequential(
#             nn.Conv2d(128, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 1, kernel_size=3, padding=1),
#             nn.Sigmoid()  # Normalize output
#         )
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = F.interpolate(x, scale_factor=2)  # Upsample
#         x = self.decoder(x)
#         return x


class DeclipCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=15, padding=7)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        x = self.decoder(x)
        return x



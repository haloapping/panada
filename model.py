from typing import Any
from torch import nn


class Model(nn.Module):
    def __init__(self, num_channels: int=1, num_classes: int=23) -> None:
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5408, num_classes),
            nn.Softmax(dim=1),
        )
        
    def forward(self, x) -> Any:
        features = self.features(x)
        probs = self.classifier(features)
        
        return probs
import torch.nn as nn
import torch
from ai_models.backbones_defunct import DenseNetCBAMBackbone
from ai_models.classifier import ClassifierHead

class DenseNetCBAM(nn.Module):
    def __init__(self, num_classes=14, dropout=0.3):
        super().__init__()
        self.backbone = DenseNetCBAMBackbone()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        in_features = 1024
        self.classifier_head = ClassifierHead(
            in_features=in_features,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(self, x):

        x = self.backbone(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier_head(x)

        return x
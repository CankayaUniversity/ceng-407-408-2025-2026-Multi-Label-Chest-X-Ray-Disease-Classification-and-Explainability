from torchvision.models.densenet import _DenseBlock, DenseNet121_Weights, densenet121
import torch.nn as nn
from ai_models.CBAM import cbam

class DenseNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        base = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

        new_features = []

        for module in base.features.children():
            new_features.append(module)

            if isinstance(module, _DenseBlock):
                channels = module.num_output_features
                new_features.append(cbam(in_channels=channels))

        self.features = nn.Sequential(*new_features)

    def forward(self, x, return_features=False):
        features = [] # for gradCAM
        for module in self.features:
            x = module(x)
            if isinstance(module, cbam):
                features.append(x)

        if return_features:
            return x, features
        return x

import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, in_features=1024, num_classes=14, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
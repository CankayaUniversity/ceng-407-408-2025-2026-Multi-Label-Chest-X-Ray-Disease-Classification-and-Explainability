
import torch.nn as nn
import torch



class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction=16): #--try reduction=32 later for les param count--
        super().__init__()

        mid = max(input_channels// reduction, 1)

        # small MLP
        self.mlp = nn.Sequential(
            # bias=False to reduce parameter count
            nn.Linear(input_channels,mid,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, input_channels, bias=False)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        batchSize, Channels, _, _ = x.shape

        avg = self.avg_pool(x).view(batchSize, Channels)
        max = self.max_pool(x).view(batchSize, Channels)

        avg_out = self.mlp(avg)
        max_out = self.mlp(max)

        out = avg_out + max_out
        out = self.sigmoid(out).view(batchSize, Channels, 1, 1)
        # multiply the current features by the sigmoid of avg amd max pool
        return x * out




class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7" # has to be either 3 or 7 always
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # along channel: compute max and avg
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        concat = torch.cat([max_pool, avg_pool], dim=1)
        out = self.conv(concat)
        out = self.sigmoid(out)
        # multiply the current features by the sigmoid of the convolution of the avg and max pool concatenated
        return x * out




class cbam(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x






#Multi-file CNN structure !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
## Pooling Layers

import torch
import torch.nn as nn

class StatsPooling(nn.Module):
    def __init__(self):
        super(StatsPooling, self).__init__()

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = torch.sqrt((x - mean).pow(2).mean(-1) + 1e-5)
        return torch.cat([mean.squeeze(-1), var], -1)


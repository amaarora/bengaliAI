import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F

class FinalLayers(nn.Module):
    def __init__(self):
        super(FinalLayers, self).__init__()
        self.l0 = nn.Linear(512, 168)
        self.l1 = nn.Linear(512, 11)
        self.l2 = nn.Linear(512, 7)

    def forward(self, x):
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        return l0, l1, l2


class Resnet34(nn.Module):
    def __init__(self, pretrained=False):
        super(Resnet34, self).__init__()
        model = models.resnet34(pretrained)
        self.features = nn.Sequential(
            *list(model.children())[:-1]
            )
        self.linear_layers = FinalLayers()
        
    def forward(self, x):
        bs, _, _, _ = x.shape
        x  = self.features(x)
        x  = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0, l1, l2 = self.linear_layers(x)
        return l0, l1, l2






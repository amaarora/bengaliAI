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
        features = nn.Sequential(
            *list(model.children())[:-1]
            )
        n = len(features)//2
        self.initial_layers = features[:4]
        self.middle_layers  = features[4:7]
        self.later_layers   = features[7:]
        self.linear_layers = FinalLayers()
        
    def forward(self, x):
        bs, _, _, _ = x.shape
        x  = self.initial_layers(x)
        x  = self.middle_layers(x)
        x  = self.later_layers(x)

        x  = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0, l1, l2 = self.linear_layers(x)
        return l0, l1, l2


class MiniNet(nn.Module):
    def __init__(self, n_out):
        super(MiniNet, self).__init__()
        self.l1    = nn.Linear(2048, 512)
        self.l2    = nn.Linear(512, n_out)
    
    def forward(self):
        return self.l2(F.relu(self.l1))


class FinalLayersResnet50(nn.Module):
    def __init__(self):
        super(FinalLayersResnet50, self).__init__()
        self.l0 = MiniNet(168)
        self.l1 = MiniNet(11)
        self.l2 = MiniNet(7)

    def forward(self, x):
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        return l0, l1, l2


class Resnet50(nn.Module):
    def __init__(self, pretrained=False):
        super(Resnet50, self).__init__()
        model = models.resnet50(pretrained)
        features = nn.Sequential(
            *list(model.children())[:-1]
            )
        
        self.initial_layers = features[:4]
        self.middle_layers  = features[4:7]
        self.later_layers   = features[7:]
        self.linear_layers = FinalLayersResnet50()
        
    def forward(self, x):
        bs, _, _, _ = x.shape
        x  = self.initial_layers(x)
        x  = self.middle_layers(x)
        x  = self.later_layers(x)

        x  = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0, l1, l2 = self.linear_layers(x)
        return l0, l1, l2





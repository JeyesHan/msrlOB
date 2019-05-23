import torch
from torch import nn
from torch.nn import init
import torchvision

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

class Q_zoom(nn.Module):
    "caculate expected cumulative reward of (st,ai)"
    def __init__(self):
        super().__init__()
        self.his_actions = 4
        self.num_actions = 6
        self.zoom_net = nn.Sequential(
            nn.Linear(7*7*512+self.his_actions*self.num_actions, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024,self.num_actions),
            nn.Softmax(dim=1)
        )
        self.zoom_net.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.zoom_net(x)
        return x


class Q_refine(nn.Module):
    "caculate expected cumulative reward of (st,ai)"

    def __init__(self):
        super().__init__()
        self.his_actions = 4
        self.num_actions = 6
        self.refine = nn.Sequential(
            nn.Linear(7 * 7 * 512 + self.his_actions * self.num_actions, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, self.num_actions),
            nn.Softmax(dim=1)
        )
        self.refine.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.fefine(x)
        return x

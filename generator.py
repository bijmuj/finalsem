import torch
import torch.nn as nn

from basic_vsr import BasicVSR
from ifnet import IFNet


class BasicGenerator(nn.Module):
    def __init__(self, spynet_path):
        super(Generator, self).__init__()
        self.ifnet = IFNet()
        self.basic = BasicVSR(num_block=30, num_feat=64, spynet_path=spynet_path)
    
    def forward(self, x):
        

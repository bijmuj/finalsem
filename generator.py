import torch
import torch.nn as nn

from basic_vsr import BasicVSR
from ifnet import IFNet
from spynet import SpyNet


class BasicGenerator(nn.Module):
    def __init__(self, spynet_path):
        super(BasicGenerator, self).__init__()
        self.ifnet = IFNet()
        self.spynet = SpyNet(spynet_path)
        self.basic = BasicVSR(self.spynet, num_block=30, num_feat=64)

    def forward(self, x):
        """Forward call for BasicGenerator.

        Args:
            x (torch.tensor): Input tensor of shape (b, n, c, h, w).
                    c should be 3.
                    n should be 4 for Vimeo90k septuplet dataset.

        Returns:
            torch.tensor: Output tensor of shape (b, n + ceil(n/2) - 1, c, h, w).
        """
        upscaled = self.basic(x)
        # print(upscaled.size())
        b, n, c, h, w = upscaled.size()
        upscaled_x1 = upscaled[:, :-1].reshape(-1, c, h, w)
        upscaled_x2 = upscaled[:, 1:].reshape(-1, c, h, w)

        upscaled_inter = self.ifnet(upscaled_x1, upscaled_x2, flow_only=False)
        upscaled_inter = upscaled_inter.view(b, n - 1, c, h, w)

        final_tensor = torch.zeros((b, 2 * n - 1, c, h, w))
        final_tensor[:, ::2] = upscaled
        final_tensor[:, 1::2] = upscaled_inter

        return final_tensor


# class MergedGenerator(nn.Module):
#     def __init__(self):
#         super(MergedGenerator, self).__init__()
#         self.ifnet = IFNet()
#         self.basic = BasicVSR(self.ifnet)

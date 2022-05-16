import torch
import torch.nn as nn

from basic_vsr import BasicVSR
from ifnet import IFNet
from model_utils import BicubicUpsampler, warp
from spynet import SpyNet


class BasicToRifeGenerator(nn.Module):
    def __init__(self, spynet_path: str):
        super(BasicToRifeGenerator, self).__init__()
        self.ifnet = IFNet()
        self.spynet = SpyNet(spynet_path)
        self.basic = BasicVSR(self.spynet, num_block=30, num_feat=64)

    def forward(self, x: torch.tensor) -> torch.tensor:
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

        _, _, upscaled_inter, _, _, _ = self.ifnet(upscaled_x1, upscaled_x2, flow_only=False)
        upscaled_inter = upscaled_inter[2].view(b, n - 1, c, h, w)

        final_tensor = torch.zeros((b, 2 * n - 1, c, h, w))
        final_tensor[:, ::2] = upscaled
        final_tensor[:, 1::2] = upscaled_inter

        return final_tensor


class RifeToBasicGenerator(nn.Module):
    def __init__(self, spynet_path: str):
        super(RifeToBasicGenerator, self).__init__()
        self.ifnet = IFNet()
        self.spynet = SpyNet(spynet_path)
        self.basic = BasicVSR(self.spynet, num_block=30, num_feat=64)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward call for UpscalingGenerator.

        Args:
            x (torch.tensor): Input frames of shape (b, n, c, h, w).

        Returns:
            torch.tensor: Output frames of shape (b, 2 * n - 1, c, h, w).
        """
        b, c, n, h, w = x.size()
        x1 = x[:, :-1].reshape(-1, c, h, w)
        x2 = x[:, 1:].reshape(-1, c, h, w)

        _, _, inter, _, _, _ = self.ifnet(x1, x2, flow_only=False)
        inter = inter[2].view(b, n - 1, c, h, w)

        mixed = torch.zeros((b, 2 * n - 1, c, h, w))
        mixed[:, ::2] = x
        mixed[:, 1::2] = inter

        return self.basic(mixed)


class UpscalingGenerator(nn.Module):
    def __init__(self, scale: int = 4):
        super(UpscalingGenerator, self).__init__()
        self.ifnet = IFNet()
        self.basic = BasicVSR(self.ifnet, num_block=30, num_feat=64)
        self.scale = scale
        self.upsampler = BicubicUpsampler(scale)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward call for UpscalingGenerator.

        Args:
            x (torch.tensor): Input frames in shape (b, n, c, h, w).

        Returns:
            torch.tensor: Output tensor.
        """
        b, n, c, h, w = x.size()
        x1 = x[:, :-1].reshape(-1, c, h, w)
        x2 = x[:, 1:].reshape(-1, c, h, w)

        flows, mask, _, _, _, _ = self.ifnet(x1, x2, flow_only=False)
        flows = flows[2]

        upscaled_flows = self.upsampler(flows)
        upscaled_masks = self.upsampler(mask)

        flows_forward = flows[:, :2].view(b, n - 1, 2, h, w)
        flows_backward = flows[:, 2:4].view(b, n - 1, 2, h, w)
        upscaled = self.basic(x, flows_forward, flows_backward)

        upscaled_a1 = upscaled[:, :-1].reshape(-1, 3, h * self.scale, w * self.scale)
        upscaled_a2 = upscaled[:, 1:].reshape(-1, 3, h * self.scale, w * self.scale)

        upscaled_warp1 = warp(upscaled_a1, upscaled_flows[:, :2].permute(0, 2, 3, 1))
        upscaled_warp2 = warp(upscaled_a2, upscaled_flows[:, 2:4].permute(0, 2, 3, 1))

        upscaled_merge = upscaled_warp1 * upscaled_masks + upscaled_warp2 * (1 - upscaled_masks)
        upscaled_merge = upscaled_merge.reshape(b, n - 1, c, h * self.scale, w * self.scale)

        mixed = torch.zeros((b, 2 * n - 1, c, h * self.scale, w * self.scale))
        mixed[:, ::2] = upscaled.reshape(b, n, c, h * self.scale, w * self.scale)
        mixed[:, 1::2] = upscaled_merge

        return mixed


# class MergedGenerator(nn.Module):
#     def __init__(self):
#         b, n, c, h, w = x.size()
#         x1 = x[:, :-1].reshape(-1, c, h, w)
#         x2 = x[:, 1:].reshape(-1, c, h, w)

#         flows, mask, _, _, _, _ = self.ifnet(x1, x2, flows_only=False)
#         flows = flows[2]

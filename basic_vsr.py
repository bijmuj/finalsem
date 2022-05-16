import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import default_init_weights, make_layer, warp


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.
    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|
    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.
    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch),
        )

    def forward(self, fea):
        return self.main(fea)


class BasicVSR(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.
    Args:
        flow_net (nn.Module): Flow estimation network.
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
    """

    def __init__(self, flow_net, num_feat=64, num_block=15):
        super().__init__()
        self.num_feat = num_feat

        # alignment
        self.flownet = flow_net
        self.flownet_name = flow_net.__class__.__name__

        # propagation
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)
        self.forward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        # reconstruction
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1].reshape(-1, 3, h, w)
        x_2 = x[:, 1:].reshape(-1, 3, h, w)

        if self.flownet_name == "SpyNet":
            flows_backward = self.flownet(x_1, x_2).view(b, n - 1, 2, h, w)
            flows_forward = self.flownet(x_2, x_1).view(b, n - 1, 2, h, w)

        else:
            flows = self.flownet(x_1, x_2)
            flows_forward = flows[:, :2].view(b, n - 1, 2, h, w)
            flows_backward = flows[:, 2:4].view(b, n - 1, 2, h, w)
        return flows_forward, flows_backward

    def forward(self, x, flows_forward=None, flows_backward=None):
        """Forward function of BasicVSR.
        Args:
            x (torch.tensor): Input frames with shape (b, n, c, h, w).
                    c is channels per frame * num of frames.
            flows_forward (torch.tensor): Forward optical flow estimations of shape
                    (b, n-1, 2, h, w). Default None.
            flows_forward (torch.tensor): Backward optical flow estimations of shape
                    (b, n-1, 2, h, w). Default None.
        """
        if flows_forward is None or flows_backward is None:
            flows_forward, flows_backward = self.get_flow(x)
        b, n, c, h, w = x.size()

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # upsample
            out = torch.cat([out_l[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode="bilinear", align_corners=False)
            out += base
            out_l[i] = out

        return torch.stack(out_l, dim=1)

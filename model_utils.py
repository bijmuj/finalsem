import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.
    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


def warp(x, flow, interp_mode="bilinear", padding_mode="zeros", align_corners=True):
    """Warp an image or feature map with optical flow.
    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.
    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(
        x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners
    )

    # TODO, what if align_corners=False
    return output


class BicubicUpsampler(nn.Module):
    """Bicubic upsampling function with similar behavior to that in TecoGAN-Tensorflow
    Note:
        This function is different from torch.nn.functional.interpolate and matlab's imresize
        in terms of the bicubic kernel and the sampling strategy
    References:
        http://verona.fi-p.unam.mx/boris/practicas/CubConvInterp.pdf
        https://stackoverflow.com/questions/26823140/imresize-trying-to-understand-the-bicubic-interpolation
    """

    def __init__(self, scale_factor, a=-0.75):
        super(BicubicUpsampler, self).__init__()

        # calculate weights (according to Eq.(6) in the reference paper)
        cubic = torch.FloatTensor(
            [
                [0, a, -2 * a, a],
                [1, 0, -(a + 3), a + 2],
                [0, -a, (2 * a + 3), -(a + 2)],
                [0, 0, a, -a],
            ]
        )

        kernels = [
            torch.matmul(cubic, torch.FloatTensor([1, s, s ** 2, s ** 3]))
            for s in [1.0 * d / scale_factor for d in range(scale_factor)]
        ]  # s = x - floor(x)

        # register parameters
        self.scale_factor = scale_factor
        self.register_buffer("kernels", torch.stack(kernels))  # size: (f, 4)

    def forward(self, input):
        n, c, h, w = input.size()
        f = self.scale_factor

        # merge n&c
        input = input.reshape(n * c, 1, h, w)

        # pad input (left, right, top, bottom)
        input = F.pad(input, (1, 2, 1, 2), mode="replicate")

        # calculate output (vertical expansion)
        kernel_h = self.kernels.view(f, 1, 4, 1)
        output = F.conv2d(input, kernel_h, stride=1, padding=0)
        output = output.permute(0, 2, 1, 3).reshape(n * c, 1, f * h, w + 3)

        # calculate output (horizontal expansion)
        kernel_w = self.kernels.view(f, 1, 1, 4)
        output = F.conv2d(output, kernel_w, stride=1, padding=0)
        output = output.permute(0, 2, 3, 1).reshape(n * c, 1, f * h, f * w)

        # split n&c
        output = output.reshape(n, c, f * h, f * w)

        return output

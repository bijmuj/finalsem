import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import warp


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.PReLU(out_planes),
    )


def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        ),
        nn.BatchNorm2d(out_planes),
        nn.PReLU(out_planes),
    )


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(self, x, flow, scale):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = (
                F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
                * 1.0
                / scale
            )
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask


class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block_tea = IFBlock(10 + 4, c=90)
        self.blocks = [IFBlock(6, c=90), IFBlock(7 + 4, c=90), IFBlock(7 + 4, c=90)]

    def forward(self, img0, img1, gt=None, flow_only=True, scale_list=[4, 2, 1]):
        if flow_only:
            return self.get_flow_only(img0, img1)
        else:
            return self.get_all(img0, img1, gt)

    def get_flow_only(self, img0, img1, scale_list=[4, 2, 1]):
        flow = None
        warped_img0 = img0
        warped_img1 = img1

        for i in range(3):
            if flow is None:
                flow, mask = self.blocks[i](torch.cat((img0, img1), 1), None, scale=scale_list[i])
            else:
                flow_d, mask_d = self.blocks[i](
                    torch.cat((img0, img1, warped_img0, warped_img1, mask), 1),
                    flow,
                    scale=scale_list[i],
                )
                flow = flow + flow_d
                mask = mask + mask_d
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])

        return flow

    def get_all(self, img0, img1, gt, scale_list=[4, 2, 1]):
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        loss_distill = 0

        for i in range(3):
            if flow is not None:
                flow_d, mask_d = self.blocks[i](
                    torch.cat((img0, img1, warped_img0, warped_img1, mask), 1),
                    flow,
                    scale=scale_list[i],
                )
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = self.blocks[i](torch.cat((img0, img1), 1), None, scale=scale_list[i])
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)

        if gt.shape[1] == 3:
            flow_d, mask_d = self.block_tea(
                torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow, scale=1
            )
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (
                1 - mask_teacher
            )
        else:
            flow_teacher = None
            merged_teacher = None

        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = (
                    (
                        (merged[i] - gt).abs().mean(1, True)
                        > (merged_teacher - gt).abs().mean(1, True) + 0.01
                    )
                    .float()
                    .detach()
                )
                loss_distill += (
                    ((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask
                ).mean()

        # c0 = self.contextnet(img0, flow[:, :2])
        # c1 = self.contextnet(img1, flow[:, 2:4])
        # tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        # res = tmp[:, :3] * 2 - 1
        # merged[2] = torch.clamp(merged[2] + res, 0, 1)

        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill

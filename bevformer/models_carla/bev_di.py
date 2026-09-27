import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torchvision
from efficientnet_pytorch import EfficientNet

import math
from models_carla.ops.modules import MSDeformAttn, MSDeformAttn3D
from models_carla.models import build_model
from torchvision.models import resnet18
from typing import Dict, Tuple, Optional
from torchvision.models import resnet18, ResNet18_Weights

scene_centroid_x = 0.0
scene_centroid_y = 1.0
scene_centroid_z = 0.0

scene_centroid_py = np.array([scene_centroid_x,
                              scene_centroid_y,
                              scene_centroid_z]).reshape([1, 3])
scene_centroid = torch.from_numpy(scene_centroid_py).float()
XMIN, XMAX = -50, 50
ZMIN, ZMAX = -50, 50
YMIN, YMAX = -5, 5
bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)
EPS = 1e-6

Z, Y, X = 200, 8, 200
def pack_seqdim(tensor, B):
    shapelist = list(tensor.shape)
    B_, S = shapelist[:2]
    assert(B==B_)
    otherdims = shapelist[2:]
    tensor = torch.reshape(tensor, [B*S]+otherdims)
    return tensor
def eye_4x4(B, device='cuda'):
    rt = torch.eye(4, device=torch.device(device)).view(1,4,4).repeat([B, 1, 1])
    return rt
def unpack_seqdim(tensor, B):
    shapelist = list(tensor.shape)
    BS = shapelist[0]
    assert(BS%B==0)
    otherdims = shapelist[1:]
    S = int(BS/B)
    tensor = torch.reshape(tensor, [B,S]+otherdims)
    return tensor
def split_intrinsics(K):

    fx = K[:,0,0]
    fy = K[:,1,1]
    x0 = K[:,0,2]
    y0 = K[:,1,2]
    return fx, fy, x0, y0
def merge_intrinsics(fx, fy, x0, y0):
    B = list(fx.shape)[0]
    K = torch.zeros(B, 4, 4, dtype=torch.float32, device=fx.device)
    K[:,0,0] = fx
    K[:,1,1] = fy
    K[:,0,2] = x0
    K[:,1,2] = y0
    K[:,2,2] = 1.0
    K[:,3,3] = 1.0
    return K
def merge_rt(r, t):


    B, C, D = list(r.shape)
    B2, D2 = list(t.shape)
    assert(C==3)
    assert(D==3)
    assert(B==B2)
    assert(D2==3)
    t = t.view(B, 3)
    rt = eye_4x4(B, device=t.device)
    rt[:,:3,:3] = r
    rt[:,:3,3] = t
    return rt
def merge_rtlist(rlist, tlist):
    B, N, D, E = list(rlist.shape)
    assert(D==3)
    assert(E==3)
    B, N, F = list(tlist.shape)
    assert(F==3)

    __p = lambda x: pack_seqdim(x, B)
    __u = lambda x: unpack_seqdim(x, B)
    rlist_, tlist_ = __p(rlist), __p(tlist)
    rtlist_ = merge_rt(rlist_, tlist_)
    rtlist = __u(rtlist_)
    return rtlist
def get_camM_T_camXs(origin_T_camXs, ind=0):
    B, S = list(origin_T_camXs.shape)[0:2]
    camM_T_camXs = torch.zeros_like(origin_T_camXs)
    for b in list(range(B)):
        camM_T_origin = safe_inverse_single(origin_T_camXs[b,ind])
        for s in list(range(S)):
            camM_T_camXs[b,s] = torch.matmul(camM_T_origin, origin_T_camXs[b,s])
    return camM_T_camXs
def safe_inverse(a):
    B, _, _ = list(a.shape)
    inv = a.clone()
    r_transpose = a[:, :3, :3].transpose(1,2)

    inv[:, :3, :3] = r_transpose
    inv[:, :3, 3:4] = -torch.matmul(r_transpose, a[:, :3, 3:4])

    return inv
def split_rt_single(rt):
    r = rt[:3, :3]
    t = rt[:3, 3].view(3)
    return r, t
def safe_inverse_single(a):
    r, t = split_rt_single(a)
    t = t.view(3,1)
    r_transpose = r.t()
    inv = torch.cat([r_transpose, -torch.matmul(r_transpose, t)], 1)
    bottom_row = a[3:4, :]

    inv = torch.cat([inv, bottom_row], 0)
    return inv
def apply_4x4(RT, xyz):
    B, N, _ = list(xyz.shape)
    ones = torch.ones_like(xyz[:,:,0:1])
    xyz1 = torch.cat([xyz, ones], 2)
    xyz1_t = torch.transpose(xyz1, 1, 2)

    xyz2_t = torch.matmul(RT, xyz1_t)
    xyz2 = torch.transpose(xyz2_t, 1, 2)

    xyz2 = xyz2[:,:,:3]
    return xyz2
def matmul2(mat1, mat2):
    return torch.matmul(mat1, mat2)
def normalize_grid3d(grid_z, grid_y, grid_x, Z, Y, X, clamp_extreme=True):

    grid_z = 2.0*(grid_z / float(Z-1)) - 1.0
    grid_y = 2.0*(grid_y / float(Y-1)) - 1.0
    grid_x = 2.0*(grid_x / float(X-1)) - 1.0

    if clamp_extreme:
        grid_z = torch.clamp(grid_z, min=-2.0, max=2.0)
        grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
        grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)

    return grid_z, grid_y, grid_x
def meshgrid3d(B, Z, Y, X, stack=False, norm=False, device='cuda'):


    grid_z = torch.linspace(0.0, Z-1, Z, device=device)
    grid_z = torch.reshape(grid_z, [1, Z, 1, 1])
    grid_z = grid_z.repeat(B, 1, Y, X)

    grid_y = torch.linspace(0.0, Y-1, Y, device=device)
    grid_y = torch.reshape(grid_y, [1, 1, Y, 1])
    grid_y = grid_y.repeat(B, Z, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=device)
    grid_x = torch.reshape(grid_x, [1, 1, 1, X])
    grid_x = grid_x.repeat(B, Z, Y, 1)

    if norm:
        grid_z, grid_y, grid_x = normalize_grid3d(
            grid_z, grid_y, grid_x, Z, Y, X)

    if stack:


        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        return grid
    else:
        return grid_z, grid_y, grid_x
def gridcloud3d(B, Z, Y, X, norm=False, device='cuda'):

    grid_z, grid_y, grid_x = meshgrid3d(B, Z, Y, X, norm=norm, device=device)
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    z = torch.reshape(grid_z, [B, -1])

    xyz = torch.stack([x, y, z], dim=2)

    return xyz
def normalize_grid2d(grid_y, grid_x, Y, X, clamp_extreme=True):

    grid_y = 2.0*(grid_y / float(Y-1)) - 1.0
    grid_x = 2.0*(grid_x / float(X-1)) - 1.0

    if clamp_extreme:
        grid_y = torch.clamp(grid_y, min=-2.0, max=2.0)
        grid_x = torch.clamp(grid_x, min=-2.0, max=2.0)

    return grid_y, grid_x
def meshgrid2d(B, Y, X, stack=False, norm=False, device='cuda'):


    grid_y = torch.linspace(0.0, Y-1, Y, device=torch.device(device))
    grid_y = torch.reshape(grid_y, [1, Y, 1])
    grid_y = grid_y.repeat(B, 1, X)

    grid_x = torch.linspace(0.0, X-1, X, device=torch.device(device))
    grid_x = torch.reshape(grid_x, [1, 1, X])
    grid_x = grid_x.repeat(B, Y, 1)

    if norm:
        grid_y, grid_x = normalize_grid2d(
            grid_y, grid_x, Y, X)

    if stack:


        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid
    else:
        return grid_y, grid_x
def scale_intrinsics(K, sx, sy):
    fx, fy, x0, y0 = split_intrinsics(K)
    fx = fx*sx
    fy = fy*sy
    x0 = x0*sx
    y0 = y0*sy
    K = merge_intrinsics(fx, fy, x0, y0)
    return K

def intrinsics_from_hfov_torch(cam_w, cam_h, hfov_deg, device, dtype):
    hfov = math.radians(hfov_deg)
    fx = (cam_w / 2.0) / math.tan(hfov / 2.0)
    fy = fx
    cx = cam_w / 2.0
    cy = cam_h / 2.0
    return (torch.tensor(fx, device=device, dtype=dtype),
            torch.tensor(fy, device=device, dtype=dtype),
            torch.tensor(cx, device=device, dtype=dtype),
            torch.tensor(cy, device=device, dtype=dtype))

def euler_to_R_world_from_cam_torch(roll, pitch, yaw, device, dtype):

    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)

    Rx = torch.stack([
        torch.stack([torch.ones_like(cr), torch.zeros_like(cr), torch.zeros_like(cr)], dim=-1),
        torch.stack([torch.zeros_like(cr), cr, -sr], dim=-1),
        torch.stack([torch.zeros_like(cr), sr,  cr], dim=-1),
    ], dim=-2)

    Ry = torch.stack([
        torch.stack([ cp, torch.zeros_like(cp), sp], dim=-1),
        torch.stack([ torch.zeros_like(cp), torch.ones_like(cp), torch.zeros_like(cp)], dim=-1),
        torch.stack([-sp, torch.zeros_like(cp), cp], dim=-1),
    ], dim=-2)

    Rz = torch.stack([
        torch.stack([cy, -sy, torch.zeros_like(cy)], dim=-1),
        torch.stack([sy,  cy, torch.zeros_like(cy)], dim=-1),
        torch.stack([torch.zeros_like(cy), torch.zeros_like(cy), torch.ones_like(cy)], dim=-1),
    ], dim=-2)


    return Rz @ Ry @ Rx

@torch.no_grad()
def camlikes_to_bev_torch(
    camlikes,
    cam_pos,
    cam_rpy_deg,
    bev_h=200, bev_w=200,
    bev_scale=0.2,
    bev_cx=100.0, bev_cy=100.0,
    hfov_deg=64.6,
    mode="bilinear",
    fuse=None,
    x_front_eps=0.05,


    raw_w=400, raw_h=300,
    lb_scale=None,
    lb_pad_xy=None,
    align_corners=True,
    return_valid=False,
):


    if not torch.is_tensor(camlikes):
        camlikes = torch.as_tensor(camlikes)

    if camlikes.dim() == 3:
        camlikes = camlikes.unsqueeze(0).unsqueeze(2)
    elif camlikes.dim() == 4:
        camlikes = camlikes.unsqueeze(2)
    elif camlikes.dim() == 5:
        pass
    else:
        raise ValueError(f"camlikes must be (6,H,W) or (B,6,H,W) or (B,6,C,H,W), got {tuple(camlikes.shape)}")

    B, C6, C, H, W = camlikes.shape
    assert C6 == 6, f"expected 6 cams, got {C6}"

    device = camlikes.device
    dtype  = camlikes.dtype


    fx0, fy0, cx0, cy0 = intrinsics_from_hfov_torch(raw_w, raw_h, hfov_deg, device, dtype)


    def _to_batched_scale(scale):
        if scale is None:
            return None
        if torch.is_tensor(scale):
            s = scale.to(device=device, dtype=dtype)
        else:
            s = torch.tensor(scale, device=device, dtype=dtype)
        if s.ndim == 0:
            s = s.expand(B)
        return s

    def _to_batched_pad(pad):
        if pad is None:
            return None
        if torch.is_tensor(pad):
            p = pad.to(device=device, dtype=dtype)
        else:
            p = torch.tensor(pad, device=device, dtype=dtype)
        if p.ndim == 1 and p.numel() == 2:
            p = p.view(1, 2).expand(B, 2)
        return p

    lb_scale_b = _to_batched_scale(lb_scale)
    lb_pad_b   = _to_batched_pad(lb_pad_xy)


    if lb_scale_b is None or lb_pad_b is None:

        scale = min(W / raw_w, H / raw_h)
        new_w = raw_w * scale
        new_h = raw_h * scale
        pad_x = (W - new_w) * 0.5
        pad_y = (H - new_h) * 0.5
        lb_scale_b = torch.full((B,), float(scale), device=device, dtype=dtype)
        lb_pad_b   = torch.tensor([pad_x, pad_y], device=device, dtype=dtype).view(1,2).expand(B,2)

    scale_b = lb_scale_b.view(B, 1, 1, 1)
    pad_x_b = lb_pad_b[:, 0].view(B, 1, 1, 1)
    pad_y_b = lb_pad_b[:, 1].view(B, 1, 1, 1)

    fx = fx0 * scale_b
    fy = fy0 * scale_b
    cx = cx0 * scale_b + pad_x_b
    cy = cy0 * scale_b + pad_y_b


    vs = torch.arange(bev_h, device=device, dtype=dtype)
    us = torch.arange(bev_w, device=device, dtype=dtype)
    vv, uu = torch.meshgrid(vs, us, indexing="ij")

    Xw = (bev_cy - vv) * bev_scale
    Yw = (uu - bev_cx) * bev_scale
    Zw = torch.zeros_like(Xw)
    Pw = torch.stack([Xw, Yw, Zw], dim=-1)


    cam_pos_t = torch.as_tensor(cam_pos, device=device, dtype=dtype).view(6, 3)
    cam_rpy_t = torch.as_tensor(cam_rpy_deg, device=device, dtype=dtype).view(6, 3)

    rpy = cam_rpy_t * (math.pi / 180.0)
    roll, pitch, yaw = rpy[:, 0], rpy[:, 1], rpy[:, 2]

    R_wc = euler_to_R_world_from_cam_torch(roll, pitch, yaw, device, dtype)
    R_cw = R_wc.transpose(-1, -2)

    Pw6 = Pw.unsqueeze(0).expand(6, -1, -1, -1)
    Cw6 = cam_pos_t[:, None, None, :]
    delta = (Pw6 - Cw6)


    delta_col = delta.unsqueeze(-1)
    Pc = (R_cw[:, None, None, :, :] @ delta_col).squeeze(-1)

    Xc, Yc, Zc = Pc[..., 0], Pc[..., 1], Pc[..., 2]


    eps = 1e-6
    ratio_u = (Yc / (Xc + eps)).unsqueeze(0)
    ratio_v = (Zc / (Xc + eps)).unsqueeze(0)

    u_img = cx + fx * ratio_u
    v_img = cy - fy * ratio_v

    Xc_b = Xc.unsqueeze(0)
    valid = (Xc_b > x_front_eps) & (u_img >= 0) & (u_img <= (W - 1)) & (v_img >= 0) & (v_img <= (H - 1))


    x_norm = (u_img / (W - 1)) * 2 - 1
    y_norm = (v_img / (H - 1)) * 2 - 1
    grid = torch.stack([x_norm, y_norm], dim=-1)


    inp = camlikes.reshape(B * 6, C, H, W).contiguous()
    grid_b = grid.reshape(B * 6, bev_h, bev_w, 2).contiguous()

    bev_out = F.grid_sample(
        inp, grid_b,
        mode=mode,
        padding_mode="zeros",
        align_corners=align_corners,
    )

    bev_out = bev_out.view(B, 6, C, bev_h, bev_w)


    bev_out = bev_out * valid.unsqueeze(2).to(dtype)

    if fuse is None:
        return (bev_out, valid) if return_valid else bev_out

    if fuse == "max":
        bev_fused = bev_out.max(dim=1).values
    elif fuse == "mean":
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1).to(dtype)
        bev_fused = bev_out.sum(dim=1) / denom
    else:
        raise ValueError("fuse must be None, 'max', or 'mean'")

    return (bev_out, bev_fused, valid) if return_valid else (bev_out, bev_fused)


def GN(C, max_groups=32):

    for g in range(min(max_groups, C), 0, -1):
        if C % g == 0:
            return nn.GroupNorm(g, C)
    return nn.GroupNorm(1, C)
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            GN(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            GN(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)

        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class BevEncode_effi_B0(nn.Module):
    def __init__(self, inC):
        super(BevEncode_effi_B0, self).__init__()

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")
        self.trunk._conv_stem = nn.Conv2d(inC, 32, kernel_size=3, padding=1, bias=False)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(112, 64, kernel_size=3, padding=1, bias=False),
            GN(64),
            nn.ReLU(inplace=True),)

        self.up2 = nn.Sequential(
            nn.Upsample(size=(50, 50), mode='bilinear',
                        align_corners=True),
            nn.Conv2d(320, 64, kernel_size=3, padding=1, bias=False),
            GN(64),
            nn.ReLU(inplace=True),)


        self.up1_BEV = Up(64 + 64, 64)
        self.up2_BEV = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            GN(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

        self.up1_lane = Up(64 + 64, 64)
        self.up2_lane = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            GN(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )
        self.up1_ped = Up(64 + 64, 64)
        self.up2_ped = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            GN(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )
        self.up1_veh = Up(64 + 64, 64)
        self.up2_veh = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            GN(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )
        self.up1_stop = Up(64 + 64, 64)
        self.up2_stop = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            GN(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

    def get_eff_depth(self, x):
        endpoints = []

        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x


        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints.append(prev_x)
            prev_x = x


        endpoints.append(x)
        del endpoints[0]
        return endpoints

    def forward(self, x):
        pyramid = self.get_eff_depth(x)

        ups=self.up1(pyramid[2])
        ups2=self.up2(pyramid[3])

        road = self.up1_BEV(ups2, ups)
        road = self.up2_BEV(road)

        lane = self.up1_lane(ups2, ups)
        lane = self.up2_lane(lane)

        veh = self.up1_veh(ups2, ups)
        veh = self.up2_veh(veh)

        ped = self.up1_ped(ups2, ups)
        ped = self.up2_ped(ped)

        stop = self.up1_stop(ups2, ups)
        stop = self.up2_stop(stop)


        return road, lane, veh, ped, stop
class FPN(nn.Module):
    def __init__(self, in_channels=[128, 256, 512], out_channels=256):
        super().__init__()

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels
        ])

    def forward(self, x_list):

        laterals = [conv(x) for conv, x in zip(self.lateral_convs, x_list)]

        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(laterals[i], size=laterals[i-1].shape[-2:], mode='nearest')
            laterals[i-1] = laterals[i-1] + upsampled

        outs = [conv(x) for conv, x in zip(self.fpn_convs, laterals)]

        return outs

def warp_with_flow(x, flow_norm):

    B, C, H, W = x.shape
    base = F.affine_grid(torch.eye(2,3, device=x.device).unsqueeze(0).repeat(B,1,1),
                         size=x.size(), align_corners=True)
    grid = base + flow_norm.permute(0,2,3,1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

class MicroAlignFuse(nn.Module):
    def __init__(self, C, max_disp_px=2, hidden=64):
        super().__init__()
        self.max_disp_px = max_disp_px

        self.flow_net = nn.Sequential(
            nn.Conv2d(C*2, hidden, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(hidden, 2, 1)
        )

        self.gate = nn.Sequential(
            nn.Conv2d(C*2, hidden, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(hidden, C, 1),
            nn.Sigmoid()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(C*2, hidden, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(hidden, C, 1)
        )

    def forward(self, bev_now, bev_hist_aligned):

        B, C, H, W = bev_now.shape
        x = torch.cat([bev_now, bev_hist_aligned], dim=1)

        flow_px = torch.tanh(self.flow_net(x)) * self.max_disp_px

        flow_norm = torch.zeros_like(flow_px)
        flow_norm[:, 0] = 2.0 * flow_px[:, 0] / max(W - 1, 1)
        flow_norm[:, 1] = 2.0 * flow_px[:, 1] / max(H - 1, 1)

        bev_hist_micro = warp_with_flow(bev_hist_aligned, flow_norm)

        y = torch.cat([bev_now, bev_hist_micro], dim=1)
        g = self.gate(y)
        delta = self.fuse(y)
        out = bev_now + g * delta
        return out, bev_hist_micro, flow_px


class Encoder(nn.Module):
    def __init__(self, data_conf):
        super().__init__()

        self.camencode = build_model(data_conf)
        checkpoint = torch.load('internimage_t_1k_224.pth', map_location='cpu')
        self.camencode.load_state_dict(checkpoint['model'], strict=False)

    def forward(self,x):

        x = self.camencode(x)

        return x
class Encoder_pv(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.C = 256


        self.camencode=Encoder(config)
        self.fpn = FPN(in_channels=[128, 256, 512], out_channels=self.C)

        self.stop_proj = nn.Conv2d(512, self.C, 1)
        self.lane_proj = nn.Conv2d(512, self.C, 1)
        self.das_proj  = nn.Conv2d(512, self.C, 1)
        self.vehicle_proj = nn.Conv2d(512, self.C, 1)
        self.walker_proj  = nn.Conv2d(512, self.C, 1)

        self.stop = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.C, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )
        self.lane = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.C, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )
        self.das = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.C, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )
        self.vehicle = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.C, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )
        self.walker = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.C, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )


        cams = {
            "left_cam":       {"pos": ( 1.53,-0.78,1.52), "rpy": (0.0, 0.0, -45.0)},
            "front_cam":      {"pos": ( 1.7,  0.0, 1.54), "rpy": (0.0, 0.0,   0.0)},
            "right_cam":      {"pos": ( 1.53, 0.78,1.52), "rpy": (0.0, 0.0,  45.0)},
            "rear_left_cam":  {"pos": (-0.78,-0.78,1.51), "rpy": (0.0, 0.0,-135.0)},
            "rear_cam":       {"pos": (-1.0,  0.0, 1.54), "rpy": (0.0, 0.0, 180.0)},
            "rear_right_cam": {"pos": (-0.78, 0.78,1.51), "rpy": (0.0, 0.0, 135.0)},
        }
        self.cam_pos = np.stack([cfg["pos"] for _, cfg in cams.items()], axis=0).astype(np.float32)
        self.cam_rpy = np.stack([cfg["rpy"] for _, cfg in cams.items()], axis=0).astype(np.float32)
        self.micro_align_fuse=MicroAlignFuse(5)


    def history(self, x):
        B, N, C, H, W = x.shape
        x = x.view(B*N, C, H, W)

        feats = self.camencode(x)
        fpn_outs = self.fpn(feats[1:])

        p3 = feats[-1]

        stop   = self.stop(self.stop_proj(p3))
        lane   = self.lane(self.lane_proj(p3))
        das    = self.das(self.das_proj(p3))
        vehicle= self.vehicle(self.vehicle_proj(p3))
        walker = self.walker(self.walker_proj(p3))

        pv_stop = F.interpolate(stop, size=(H, W), mode="bilinear", align_corners=True).view(B, 6, 1, H, W)
        pv_lane = F.interpolate(lane, size=(H, W), mode="bilinear", align_corners=True).view(B, 6, 1, H, W)
        pv_road = F.interpolate(das,  size=(H, W), mode="bilinear", align_corners=True).view(B, 6, 1, H, W)
        pv_veh  = F.interpolate(vehicle, size=(H, W), mode="bilinear", align_corners=True).view(B, 6, 1, H, W)
        pv_ped  = F.interpolate(walker,  size=(H, W), mode="bilinear", align_corners=True).view(B, 6, 1, H, W)


        _, bev_road = camlikes_to_bev_torch(pv_road,cam_pos=self.cam_pos,cam_rpy_deg=self.cam_rpy, mode="bilinear", fuse="max")
        _, bev_lane = camlikes_to_bev_torch(pv_lane,cam_pos=self.cam_pos,cam_rpy_deg=self.cam_rpy, mode="bilinear", fuse="max")
        _, bev_veh = camlikes_to_bev_torch(pv_veh,cam_pos=self.cam_pos,cam_rpy_deg=self.cam_rpy, mode="bilinear", fuse="max")
        _, bev_ped = camlikes_to_bev_torch(pv_ped,cam_pos=self.cam_pos,cam_rpy_deg=self.cam_rpy, mode="bilinear", fuse="max")
        _, bev_stop = camlikes_to_bev_torch(pv_stop,cam_pos=self.cam_pos,cam_rpy_deg=self.cam_rpy, mode="bilinear", fuse="max")


        return bev_road,bev_lane,bev_veh,bev_ped,bev_stop


    def forward(self, x,history_bev):
        B, N, C, H, W = x.shape
        x = x.view(B*N, C, H, W)

        feats = self.camencode(x)
        fpn_outs = self.fpn(feats[1:])

        p3 = feats[-1]
        stop   = self.stop(self.stop_proj(p3))
        lane   = self.lane(self.lane_proj(p3))
        das    = self.das(self.das_proj(p3))
        vehicle= self.vehicle(self.vehicle_proj(p3))
        walker = self.walker(self.walker_proj(p3))

        pv_stop = F.interpolate(stop, size=(H, W), mode="bilinear", align_corners=True).view(B, 6, 1, H, W)
        pv_lane = F.interpolate(lane, size=(H, W), mode="bilinear", align_corners=True).view(B, 6, 1, H, W)
        pv_road = F.interpolate(das,  size=(H, W), mode="bilinear", align_corners=True).view(B, 6, 1, H, W)
        pv_veh  = F.interpolate(vehicle, size=(H, W), mode="bilinear", align_corners=True).view(B, 6, 1, H, W)
        pv_ped  = F.interpolate(walker,  size=(H, W), mode="bilinear", align_corners=True).view(B, 6, 1, H, W)


        _, bev_road = camlikes_to_bev_torch(pv_road,cam_pos=self.cam_pos,cam_rpy_deg=self.cam_rpy, mode="bilinear", fuse="max")
        _, bev_lane = camlikes_to_bev_torch(pv_lane,cam_pos=self.cam_pos,cam_rpy_deg=self.cam_rpy, mode="bilinear", fuse="max")
        _, bev_veh = camlikes_to_bev_torch(pv_veh,cam_pos=self.cam_pos,cam_rpy_deg=self.cam_rpy, mode="bilinear", fuse="max")
        _, bev_ped = camlikes_to_bev_torch(pv_ped,cam_pos=self.cam_pos,cam_rpy_deg=self.cam_rpy, mode="bilinear", fuse="max")
        _, bev_stop = camlikes_to_bev_torch(pv_stop,cam_pos=self.cam_pos,cam_rpy_deg=self.cam_rpy, mode="bilinear", fuse="max")


        bev_now = torch.cat([bev_road, bev_lane, bev_veh, bev_ped, bev_stop], dim=1)

        bev_fused, bev_hist_micro, flow_px = self.micro_align_fuse(bev_now, history_bev)


        return bev_fused

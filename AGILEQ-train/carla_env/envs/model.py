import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

import numpy as np

def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))


        ctx.save_for_backward(kept)


        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx
def scatter_sum_replacement(src, index, dim=0, dim_size=None):
    if dim_size is None:
        dim_size = int(index.max().item()) + 1

    if index.ndim < src.ndim:
        index = index.unsqueeze(-1)

    out = torch.zeros(*src.shape[:dim], dim_size, *src.shape[dim+1:], dtype=src.dtype, device=src.device)
    return out.scatter_add(dim, index, src)

def scatter_mean_replacement(src, index, dim=0, dim_size=None):
    sum_ = scatter_sum_replacement(src, index, dim, dim_size)
    count = scatter_sum_replacement(torch.ones_like(src), index, dim, dim_size)
    count = torch.clamp(count, min=1)
    return sum_ / count

def raval_index(coords, dims):
    dims = torch.cat((dims, torch.ones(1, device=dims.device)), dim=0)[1:]
    dims = torch.flip(dims, dims=[0])
    dims = torch.cumprod(dims, dim=0) / dims[0]
    multiplier = torch.flip(dims, dims=[0])
    indices = torch.sum(coords * multiplier, dim=1)
    return indices


def points_to_voxels(
  points_xyz,
  points_mask,
  grid_range_x,
  grid_range_y,
  grid_range_z
):
    batch_size, num_points, _ = points_xyz.shape
    voxel_size_x = grid_range_x[2]
    voxel_size_y = grid_range_y[2]
    voxel_size_z = grid_range_z[2]
    grid_size = np.asarray([
        (grid_range_x[1]-grid_range_x[0]) / voxel_size_x,
        (grid_range_y[1]-grid_range_y[0]) / voxel_size_y,
        (grid_range_z[1]-grid_range_z[0]) / voxel_size_z
    ]).astype('int32')
    voxel_size = np.asarray([voxel_size_x, voxel_size_y, voxel_size_z])
    voxel_size = torch.Tensor(voxel_size).to(points_xyz.device)
    num_voxels = grid_size[0] * grid_size[1] * grid_size[2]
    grid_offset = torch.Tensor([grid_range_x[0], grid_range_y[0], grid_range_z[0]]).to(points_xyz.device)
    shifted_points_xyz = points_xyz - grid_offset
    voxel_xyz = shifted_points_xyz / voxel_size
    voxel_coords = voxel_xyz.int()
    grid_size = torch.from_numpy(grid_size).to(points_xyz.device)
    grid_size = grid_size.int()
    zeros = torch.zeros_like(grid_size)
    voxel_paddings = ((points_mask < 1.0) |
                      torch.any((voxel_coords >= grid_size) |
                                (voxel_coords < zeros), dim=-1))
    voxel_indices = raval_index(
      torch.reshape(voxel_coords, [batch_size * num_points, 3]), grid_size)
    voxel_indices = torch.reshape(voxel_indices, [batch_size, num_points])
    voxel_indices = torch.where(voxel_paddings,
                                torch.zeros_like(voxel_indices),
                                voxel_indices)
    voxel_centers = ((0.5 + voxel_coords.float()) * voxel_size + grid_offset)
    voxel_coords = torch.where(torch.unsqueeze(voxel_paddings, dim=-1),
                               torch.zeros_like(voxel_coords),
                               voxel_coords)
    voxel_xyz = torch.where(torch.unsqueeze(voxel_paddings, dim=-1),
                            torch.zeros_like(voxel_xyz),
                            voxel_xyz)
    voxel_paddings = voxel_paddings.float()

    voxel_indices = voxel_indices.long()
    points_per_voxel = scatter_sum_replacement(
        torch.ones((batch_size, num_points), dtype=voxel_coords.dtype, device=voxel_coords.device) * (1-voxel_paddings),
        voxel_indices,
        dim=1,
        dim_size=num_voxels
    )

    voxel_point_count = torch.gather(points_per_voxel,
                                     dim=1,
                                     index=voxel_indices)


    voxel_centroids = scatter_mean_replacement(
        points_xyz,
        voxel_indices,
        dim=1,
        dim_size=num_voxels)
    point_centroids = torch.gather(voxel_centroids, dim=1, index=torch.unsqueeze(voxel_indices, dim=-1).repeat(1, 1, 3))
    local_points_xyz = points_xyz - point_centroids

    result = {
        'local_points_xyz': local_points_xyz,
        'shifted_points_xyz': shifted_points_xyz,
        'point_centroids': point_centroids,
        'points_xyz': points_xyz,
        'grid_offset': grid_offset,
        'voxel_coords': voxel_coords,
        'voxel_centers': voxel_centers,
        'voxel_indices': voxel_indices,
        'voxel_paddings': voxel_paddings,
        'points_mask': 1 - voxel_paddings,
        'num_voxels': num_voxels,
        'grid_size': grid_size,
        'voxel_xyz': voxel_xyz,
        'voxel_size': voxel_size,
        'voxel_point_count': voxel_point_count,
        'points_per_voxel': points_per_voxel
    }


    return result

class PillarBlock(nn.Module):
  def __init__(self, idims=64, dims=64, num_layers=1,
               stride=1):
    super(PillarBlock, self).__init__()
    layers = []
    self.idims = idims
    self.stride = stride
    for i in range(num_layers):
      layers.append(nn.Conv2d(self.idims, dims, 3, stride=self.stride,
                              padding=1, bias=False))
      layers.append(nn.BatchNorm2d(dims))
      layers.append(nn.ReLU(inplace=True))
      self.idims = dims
      self.stride = 1
    self.layers = nn.Sequential(*layers)

  def forward(self, x):
    return self.layers(x)


class PointNet(nn.Module):
  def __init__(self, idims=64, odims=64):
    super(PointNet, self).__init__()
    self.pointnet = nn.Sequential(
      nn.Conv1d(idims, odims, kernel_size=1, bias=False),
      nn.BatchNorm1d(odims),
      nn.ReLU(inplace=True)
    )

  def forward(self, points_feature, points_mask):
    batch_size, num_points, num_dims = points_feature.shape
    points_feature = points_feature.permute(0, 2, 1)
    mask = points_mask.view(batch_size, 1, num_points)
    return self.pointnet(points_feature) * mask


class TrafficModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)
        return self.sigmoid(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)

        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class BevEncode_effi_B0(nn.Module):
    def __init__(self, inC):
        super(BevEncode_effi_B0, self).__init__()

        self.trunk = EfficientNet.from_name("efficientnet-b0")
        self.trunk._conv_stem = nn.Conv2d(inC, 32, kernel_size=3, padding=1, bias=False)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(112, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(320, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),)


        self.up1_BEV = Up(64 + 64, 64, scale_factor=2)
        self.up2_BEV = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

        self.up1_vehicle = Up(64 + 64, 64, scale_factor=2)
        self.up2_vehicle = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

        self.up1_Lane = Up(64 + 64, 64, scale_factor=2)
        self.up2_Lane = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )
        self.up1_walker = Up(64 + 64, 64, scale_factor=2)
        self.up2_walker = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

        self.up1_stopline = Up(64 + 64, 64, scale_factor=2)
        self.up2_stopline = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )
        self.up1_route = Up(64 + 64, 64, scale_factor=2)
        self.up2_route = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )
    def get_eff_feature(self, x):
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
        pyramid = self.get_eff_feature(x)

        ups=self.up1(pyramid[2])
        ups2=self.up2(pyramid[3])

        DAS = self.up1_BEV(ups2, ups)
        DAS = self.up2_BEV(DAS)

        Vehicle = self.up1_vehicle(ups2, ups)
        Vehicle = self.up2_vehicle(Vehicle)

        Lane = self.up1_Lane(ups2, ups)
        Lane = self.up2_Lane(Lane)

        walker = self.up1_walker(ups2, ups)
        walker = self.up2_walker(walker)

        route = self.up1_route(ups2, ups)
        route = self.up2_route(route)

        stopline = self.up1_stopline(ups2, ups)
        stopline = self.up2_stopline(stopline)

        return DAS, Lane, Vehicle, walker,stopline,route, ups2

class LSS_efficientnet(nn.Module):
    def __init__(self, D, C):
        super(LSS_efficientnet, self).__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b4")
        self.up1 = Up(448 + 160, 512)

        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        return new_x

    def get_eff_depth(self, x):
        endpoints = dict()


        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x


        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x


        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])

        return x

    def forward(self, x):

        x = self.get_depth_feat(x)

        return x


class bev_SEG(nn.Module):
    def __init__(self,lidar_use):
        super(bev_SEG, self).__init__()
        self.data_conf = {
            'xbound':[-20.0,20.0,0.2],
            'ybound':[-20.0,20.0,0.2],
            'zbound':[-10.0,10.0,20.0],
            'dbound':[2.0,50.0,1.0],
            'height':224,
            'width':448,

        }
        self.lidar_use=lidar_use
        dx, bx, nx = gen_dx_bx(self.data_conf['xbound'],
                                              self.data_conf['ybound'],
                                              self.data_conf['zbound'],
                                              )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 128
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = LSS_efficientnet(self.D, self.camC)
        self.use_quickcumsum = True

        if self.lidar_use:
            self.bevencode = BevEncode_effi_B0(inC=self.camC+128)
        else:
            self.bevencode = BevEncode_effi_B0(inC=self.camC)


    def create_frustum(self):

        ogfH, ogfW =self.data_conf['height'], self.data_conf['width']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.data_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)


        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):

        B, N, _ = trans.shape

        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))


        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x):

        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)
        fc_emb,x,front = self.camencode(x)


        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)


        x = x.permute(0, 1, 3, 4, 5, 2)

        return x,front,fc_emb

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W


        x = x.reshape(Nprime, C)


        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)


        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]


        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]


        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)


        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x


        final = torch.cat(final.unbind(dim=2), 1)
        final = final.permute(0, 1, 3,2)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x,x_front,fc_emb = self.get_cam_feats(x)

        x = self.voxel_pooling(geom, x)

        return x ,x_front,fc_emb


    def forward(self, x, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask):

        topdown_feature = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        if self.lidar_use:
            lidar_feature = self.pp(lidar_data, lidar_mask)
            topdown_feature = torch.cat([topdown_feature, lidar_feature], dim=1)


        DAS,Lane,Vehicle,walker,stopline,route,_ = self.bevencode(topdown_feature)


        return DAS,Lane,Vehicle,walker,stopline,route

import math
import torch
from torch import nn
from torch.nn import functional as F


class MultiArcFace(nn.Module):
    def __init__(self, in_features, centers, scale=20.0, margin=0.5):
        super().__init__()
        self.centers = centers
        self.scale = scale
        self.cos_margin = math.cos(margin)
        self.sin_margin = math.sin(margin)
        self.weight = nn.Parameter(torch.empty(centers * 2, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels=None):
        cosine = F.linear(F.normalize(features, dim=-1), F.normalize(self.weight, dim=-1))
        cosine = cosine.reshape(features.shape[0], self.centers, 2)
        if labels is not None:
            if labels.shape != (features.shape[0],) or labels.dtype != torch.long:
                raise ValueError('Light targets must be int64 class indices with shape [B]')
            sine = (1 - cosine.square()).clamp(min=1e-7, max=1).sqrt()
            margin_cosine = cosine * self.cos_margin - sine * self.sin_margin
            margin_cosine = torch.where(cosine > 0, margin_cosine, cosine)
            mask = F.one_hot(labels, num_classes=2).to(cosine.dtype).unsqueeze(1)
            cosine = mask * margin_cosine + (1 - mask) * cosine
        return cosine * self.scale


class TrafficLightHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mul_arcface = MultiArcFace(1024, config.arcface_centers, config.arcface_scale, config.arcface_margin)

    def forward(self, features, labels=None):
        return self.mul_arcface(features, labels).max(dim=1).values

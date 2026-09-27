import torch
from torch import nn
from torch.nn import functional as F


def deformable_attention(value, locations, weights, height, width):
    batch, _, heads, channels = value.shape
    queries, points = locations.shape[1], locations.shape[3]
    feature_map = value.permute(0, 2, 3, 1).reshape(batch * heads, channels, height, width)
    grid = (2 * locations - 1).permute(0, 2, 1, 3, 4).reshape(batch * heads, queries, points, 2)
    sampled = F.grid_sample(feature_map, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    weights = weights.permute(0, 2, 1, 3).reshape(batch * heads, 1, queries, points)
    output = (sampled * weights).sum(-1).reshape(batch, heads * channels, queries)
    return output.transpose(1, 2).contiguous()


class SpatialCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_heads
        self.points = config.num_sampling_points
        self.sampling_offset = nn.Linear(config.embed_dim, self.heads * self.points * 2)
        self.attention_weights = nn.Linear(config.embed_dim, self.heads * self.points)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.ref_points = nn.Parameter(torch.rand(1, config.num_queries, 2))

    def forward(self, query, value, height, width):
        batch, queries, _ = query.shape
        offsets = self.sampling_offset(query).relu().reshape(batch, queries, self.heads, self.points, 2)
        weights = self.attention_weights(query).relu().reshape(batch, queries, self.heads, self.points).softmax(-1)
        normalizer = query.new_tensor([width, height])
        locations = self.ref_points[:, :, None, None, :] + offsets / normalizer
        output = deformable_attention(value, locations, weights, height, width)
        return self.norm(query + self.dropout(output))


class TemporalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(config.embed_dim, config.num_heads, dropout=config.dropout, batch_first=True)
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, query, previous=None):
        memory = query if previous is None else previous
        output, _ = self.temporal_attn(query, memory, memory)
        return self.norm(query + output)


class TemporalSpatialEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_heads
        self.tsa = TemporalSelfAttention(config)
        self.sca = SpatialCrossAttention(config)
        self.mlp = nn.Sequential(nn.Linear(config.embed_dim, config.embed_dim), nn.ReLU(), nn.Linear(config.embed_dim, config.embed_dim))
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, queries, features):
        batch, frames, channels, height, width = features.shape
        query = queries.unsqueeze(0).expand(batch, -1, -1)
        previous = None
        for index in range(frames):
            value = features[:, index].flatten(2).transpose(1, 2).reshape(batch, height * width, self.heads, channels // self.heads)
            output = self.sca(self.tsa(query, previous), value, height, width)
            previous = self.norm(output + self.mlp(output))
        return previous

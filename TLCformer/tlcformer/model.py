import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from .attention import TemporalSpatialEncoder
from .config import ModelConfig
from .heads import TrafficLightHead


class TLCFormer(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        weights = ResNet18_Weights.DEFAULT if self.config.pretrained_backbone else None
        self.Light_img_encoder = resnet18(weights=weights)
        self.Light_img_encoder.fc = nn.Identity()
        self.Light_img_encoder.avgpool = nn.Identity()
        self.down_conv = nn.Sequential(nn.Conv2d(512, 1024, 3), nn.BatchNorm2d(1024), nn.ReLU(), nn.Conv2d(1024, 256, 3), nn.BatchNorm2d(256))
        self.mlp = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 1024))
        self.head1 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 1024))
        self.traffic_light_distance_head = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 22))
        self.query_embed = nn.Embedding(2, 256)
        self.encoder = TemporalSpatialEncoder(self.config)
        self.Light_decoder = TrafficLightHead(self.config)

    def encode_images(self, images):
        backbone = self.Light_img_encoder
        x = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(images))))
        x = backbone.layer4(backbone.layer3(backbone.layer2(backbone.layer1(x))))
        return self.down_conv(x)

    def forward(self, images, light_targets=None):
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError('Expected normalized RGB images with shape [B,T,3,H,W]')
        batch, frames, channels, height, width = images.shape
        if batch < 1 or frames < 1 or min(height, width) < 129:
            raise ValueError('B and T must be positive; H and W must be >=129')
        if not self.training and light_targets is not None:
            raise ValueError('Do not pass ground-truth labels during evaluation or inference')
        features = self.encode_images(images.reshape(batch * frames, channels, height, width))
        features = features.reshape(batch, frames, *features.shape[1:])
        tokens = self.mlp(self.encoder(self.query_embed.weight, features))
        light_features = self.head1(tokens[:, 0])
        logits = self.Light_decoder(light_features)
        loss_logits = logits if light_targets is None else self.Light_decoder(light_features, light_targets)
        return {
            'light_logits': logits,
            'light_loss_logits': loss_logits,
            'distance_logits': self.traffic_light_distance_head(tokens[:, 1]),
            'tokens': tokens,
        }

    @torch.no_grad()
    def predict(self, images):
        if self.training:
            raise RuntimeError('Call model.eval() before predict()')
        output = self(images)
        light_probs = output['light_logits'].softmax(-1)
        distance_probs = output['distance_logits'].softmax(-1)
        return {'light_probabilities': light_probs, 'light_class': light_probs.argmax(-1),
                'distance_probabilities': distance_probs, 'distance_class': distance_probs.argmax(-1)}

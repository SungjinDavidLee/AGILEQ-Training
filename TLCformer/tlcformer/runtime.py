import random
import torch


def resolve_device(name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    return device


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

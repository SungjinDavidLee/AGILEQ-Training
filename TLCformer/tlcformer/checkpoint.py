from dataclasses import replace
from pathlib import Path
import torch
from .config import Config
from .model import TLCFormer


def load_weights(model, state):
    if state and all(key.startswith('module.') for key in state):
        state = {key.removeprefix('module.'): value for key, value in state.items()}
    removed = sorted(key for key in state if key.startswith(('head2.', 'signs_decoder.')))
    state = {key: value for key, value in state.items() if key not in removed}
    model.load_state_dict(state, strict=True)
    return removed


def load_model(path, device, fallback_config=None):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError('Expected a state_dict or a TLCFormer checkpoint')
    config = Config.from_dict(payload['config']) if 'config' in payload else fallback_config
    if config is None:
        raise ValueError('A raw legacy state_dict requires --config')
    model = TLCFormer(replace(config.model, pretrained_backbone=False))
    state = payload.get('model', payload.get('state_dict', payload))
    removed = load_weights(model, state)
    return model.to(device), config, payload, removed


def save_checkpoint(path, model, optimizer, scheduler, config, epoch, best_loss, best_accuracy):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save({'format_version': 1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'config': config.to_dict(), 'epoch': epoch,
                'best_loss': best_loss, 'best_accuracy': best_accuracy}, temporary)
    temporary.replace(path)

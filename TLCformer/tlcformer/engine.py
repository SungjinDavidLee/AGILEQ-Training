import torch
from torch.nn import functional as F
from .data import FAR_CLASS


def distance_matches(predictions, targets, tolerance=0):
    near = (predictions != FAR_CLASS) & (targets != FAR_CLASS)
    return ((predictions == targets) | (near & ((predictions - targets).abs() <= tolerance)))


def compute_loss(output, batch, config):
    light = F.cross_entropy(output['light_loss_logits'], batch['light_target'])
    distance = F.cross_entropy(output['distance_logits'], batch['distance_target'])
    total = config.light_loss_weight * light + config.distance_loss_weight * distance
    return total, light, distance


def run_epoch(model, loader, device, config, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = dict.fromkeys(['loss', 'light_loss', 'distance_loss', 'light_accuracy',
                           'distance_accuracy', 'distance_within_1', 'distance_within_2'], 0.0)
    count = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(batch['images'], batch['light_target'] if training else None)
            loss, light_loss, distance_loss = compute_loss(output, batch, config)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
            batch_size = batch['images'].shape[0]
            count += batch_size
            for key, value in [('loss', loss), ('light_loss', light_loss), ('distance_loss', distance_loss)]:
                totals[key] += value.detach().item() * batch_size
            light_pred = output['light_logits'].detach().argmax(-1)
            distance_pred = output['distance_logits'].detach().argmax(-1)
            totals['light_accuracy'] += (light_pred == batch['light_target']).sum().item()
            for tolerance, key in enumerate(['distance_accuracy', 'distance_within_1', 'distance_within_2']):
                totals[key] += distance_matches(distance_pred, batch['distance_target'], tolerance).sum().item()
    if not count:
        raise ValueError('DataLoader produced no samples')
    return {key: value / count for key, value in totals.items()}

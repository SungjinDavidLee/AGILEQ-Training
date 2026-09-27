import argparse
import json
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import yaml
from tlcformer import TLCFormer
from tlcformer.checkpoint import load_model, save_checkpoint
from tlcformer.config import load_config
from tlcformer.data import TrafficLightDataset
from tlcformer.engine import run_epoch
from tlcformer.runtime import resolve_device, seed_everything


def main():
    parser = argparse.ArgumentParser(description='Train TLCFormer traffic-light state and distance recognition')
    parser.add_argument('--config', default='configs/tlcformer.yaml')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume')
    group.add_argument('--weights')
    args = parser.parse_args()
    config = load_config(args.config)
    device = resolve_device(config.train.device)
    seed_everything(config.train.seed)
    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / 'train.log', encoding='utf-8')])
    payload = None
    if args.resume or args.weights:
        model, saved_config, payload, removed = load_model(args.resume or args.weights, device, config)
        saved_model = saved_config.to_dict()['model']
        requested_model = config.to_dict()['model']
        saved_model.pop('pretrained_backbone')
        requested_model.pop('pretrained_backbone')
        if saved_model != requested_model:
            raise ValueError('Model config differs from checkpoint; use the original model settings')
        if removed:
            logging.info('Removed unused vehicle/sign-head tensors: %d', len(removed))
    else:
        model = TLCFormer(config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.train.scheduler_step_size, gamma=config.train.scheduler_gamma)
    start_epoch, best_loss, best_accuracy = 0, float('inf'), -1.0
    if args.resume:
        required = {'optimizer', 'scheduler', 'epoch', 'best_loss', 'best_accuracy'}
        if not required <= payload.keys():
            raise ValueError('Resume requires a full training checkpoint; use --weights for a legacy state_dict')
        if saved_config.data != config.data:
            raise ValueError('Resume requires the original data configuration; use --weights to start a new run')
        for key in ['learning_rate', 'weight_decay', 'scheduler_step_size', 'scheduler_gamma']:
            if getattr(saved_config.train, key) != getattr(config.train, key):
                raise ValueError(f'Resume requires the saved {key}; use --weights for a new optimizer')
        optimizer.load_state_dict(payload['optimizer'])
        scheduler.load_state_dict(payload['scheduler'])
        start_epoch = payload['epoch'] + 1
        best_loss, best_accuracy = payload['best_loss'], payload['best_accuracy']
    loaders = [DataLoader(TrafficLightDataset(config.data, split), batch_size=config.train.batch_size,
                          shuffle=training, num_workers=config.train.num_workers, pin_memory=device.type == 'cuda',
                          drop_last=False, persistent_workers=config.train.num_workers > 0)
               for split, training in [(config.data.train_split, True), (config.data.val_split, False)]]
    (output_dir / 'config.yaml').write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding='utf-8')
    logging.info('device=%s train=%d val=%d', device, len(loaders[0].dataset), len(loaders[1].dataset))
    for epoch in range(start_epoch, config.train.epochs):
        train_metrics = run_epoch(model, loaders[0], device, config.train, optimizer)
        val_metrics = run_epoch(model, loaders[1], device, config.train)
        record = {'epoch': epoch + 1, 'learning_rate': optimizer.param_groups[0]['lr'], 'train': train_metrics, 'val': val_metrics}
        logging.info(json.dumps(record, ensure_ascii=False))
        with (output_dir / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        scheduler.step()
        improved_loss = val_metrics['loss'] < best_loss
        improved_accuracy = val_metrics['light_accuracy'] > best_accuracy
        best_loss = min(best_loss, val_metrics['loss'])
        best_accuracy = max(best_accuracy, val_metrics['light_accuracy'])
        names = ['last.pt'] + (['best_loss.pt'] if improved_loss else []) + (['best_light_accuracy.pt'] if improved_accuracy else [])
        for name in names:
            save_checkpoint(output_dir / name, model, optimizer, scheduler, config, epoch, best_loss, best_accuracy)


if __name__ == '__main__':
    main()

import argparse
import json
from pathlib import Path
import torch
from tlcformer.checkpoint import load_model
from tlcformer.config import load_config
from tlcformer.data import FAR_CLASS, LIGHT_CLASSES, image_transform, load_image, natural_key
from tlcformer.runtime import resolve_device


def main():
    parser = argparse.ArgumentParser(description='Predict from one chronological front-camera sequence')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--images', nargs='+', required=True)
    parser.add_argument('--config')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output')
    args = parser.parse_args()
    device = resolve_device(args.device)
    model, config, _, _ = load_model(args.checkpoint, device, load_config(args.config) if args.config else None)
    paths = [Path(path) for path in args.images]
    if len(paths) == 1 and paths[0].is_dir():
        paths = sorted(paths[0].glob('*.png'), key=natural_key)[-config.data.sequence_length:]
    if len(paths) != config.data.sequence_length:
        raise ValueError(f'Expected {config.data.sequence_length} chronological frames, got {len(paths)}')
    transform = image_transform(config.data.image_height, config.data.image_width)
    images = torch.stack([load_image(path, transform) for path in paths]).unsqueeze(0).to(device)
    prediction = model.eval().predict(images)
    light = prediction['light_class'].item()
    distance = prediction['distance_class'].item()
    result = {'frame': str(paths[-1]), 'light_class': light, 'light_state': LIGHT_CLASSES[light],
              'light_confidence': prediction['light_probabilities'][0, light].item(),
              'distance_class': distance, 'distance_bin_m': None if distance == FAR_CLASS else distance,
              'distance_is_far_or_missing': distance == FAR_CLASS,
              'distance_confidence': prediction['distance_probabilities'][0, distance].item()}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()

import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tlcformer import TLCFormer
from tlcformer.attention import deformable_attention
from tlcformer.checkpoint import load_model, load_weights, save_checkpoint
from tlcformer.config import Config, DataConfig, ModelConfig, TrainConfig, load_config
from tlcformer.data import TrafficLightDataset, distance_to_class
from tlcformer.engine import distance_matches, run_epoch


torch.set_num_threads(2)


class TLCFormerTests(unittest.TestCase):
    def test_distance_boundaries(self):
        values = [0, 0.2, 0.5, 1.5, 19.6, 20, 20.01, 100]
        self.assertEqual([distance_to_class(v) for v in values], [21, 0, 0, 2, 20, 20, 21, 21])
        for value in [-1, float('nan'), float('inf'), '3', True]:
            with self.assertRaises(ValueError):
                distance_to_class(value)
        pred = torch.tensor([20, 21, 19, 21])
        target = torch.tensor([21, 20, 20, 21])
        self.assertEqual(distance_matches(pred, target, 2).tolist(), [False, False, True, True])

    def test_attention_coordinates_and_backward(self):
        value = torch.tensor([1., 2., 3., 4.]).reshape(1, 4, 1, 1).requires_grad_()
        locations = torch.tensor([[[[[0.25, 0.25], [0.75, 0.75]]]]], requires_grad=True)
        weights = torch.tensor([[[[0.25, 0.75]]]], requires_grad=True)
        result = deformable_attention(value, locations, weights, 2, 2)
        torch.testing.assert_close(result, torch.tensor([[[3.25]]]))
        result.sum().backward()
        for tensor in [value, locations, weights]:
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_batch_one_eval_and_state_isolation(self):
        model = TLCFormer(ModelConfig(dropout=0)).eval()
        images = torch.randn(1, 2, 3, 160, 192)
        with torch.no_grad():
            first = model(images)
            model(torch.randn_like(images))
            second = model(images)
            batched = model(torch.cat([images, images]))
        for key in first:
            torch.testing.assert_close(first[key], second[key])
            torch.testing.assert_close(first[key][0], batched[key][0], atol=1e-5, rtol=1e-4)
        self.assertEqual(first['light_logits'].shape, (1, 2))
        self.assertEqual(first['distance_logits'].shape, (1, 22))
        self.assertEqual(model.predict(images)['light_class'].shape, (1,))
        with self.assertRaises(ValueError):
            model(images, torch.tensor([1]))

    def test_training_gradients_and_optimizer(self):
        torch.manual_seed(42)
        model = TLCFormer(ModelConfig(dropout=0)).train()
        images = torch.randn(2, 2, 3, 160, 192)
        output = model(images, torch.tensor([0, 1]))
        loss = torch.nn.functional.cross_entropy(output['light_loss_logits'], torch.tensor([0, 1]))
        loss += torch.nn.functional.cross_entropy(output['distance_logits'], torch.tensor([2, 21]))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for name in ['Light_img_encoder.conv1.weight', 'encoder.tsa.temporal_attn.in_proj_weight',
                     'encoder.sca.sampling_offset.weight', 'encoder.sca.attention_weights.weight',
                     'query_embed.weight', 'Light_decoder.mul_arcface.weight', 'traffic_light_distance_head.4.weight']:
            self.assertGreater(dict(model.named_parameters())[name].grad.abs().sum().item(), 0, name)
        before = model.query_embed.weight.detach().clone()
        torch.optim.SGD(model.parameters(), lr=0.001).step()
        self.assertFalse(torch.equal(before, model.query_embed.weight))

    def test_dataset_training_validation_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / 'train' / 'Town01' / 'scene_1'
            (scene / 'front_cam_pv_rgb').mkdir(parents=True)
            (scene / 'info').mkdir()
            for frame in [1, 2, 10, 11]:
                Image.new('RGB', (24, 24), (frame, 50, 100)).save(scene / 'front_cam_pv_rgb' / f'{frame}.png')
                (scene / 'info' / f'{frame}.json').write_text(json.dumps({'traffic_light_20': frame % 2, 'traffic_light_distance': frame}))
            config = Config(model=ModelConfig(dropout=0), data=DataConfig(root=directory, sequence_length=2, skip_first=0, image_height=160, image_width=192), train=TrainConfig(batch_size=2, num_workers=0))
            dataset = TrafficLightDataset(config.data, 'train')
            self.assertEqual(len(dataset), 3)
            self.assertTrue(dataset[0]['frame_path'].endswith('2.png'))
            self.assertEqual(dataset[1]['distance_target'].item(), 10)
            self.assertEqual(dataset[0]['images'].shape, (2, 3, 160, 192))
            model = TLCFormer(config.model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
            loader = DataLoader(dataset, batch_size=2)
            train_metrics = run_epoch(model, loader, torch.device('cpu'), config.train, optimizer)
            val_metrics = run_epoch(model, loader, torch.device('cpu'), config.train)
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in train_metrics.values()))
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in val_metrics.values()))
            scheduler.step()
            checkpoint = root / 'checkpoint.pt'
            save_checkpoint(checkpoint, model, optimizer, scheduler, config, 0, val_metrics['loss'], val_metrics['light_accuracy'])
            restored, restored_config, payload, removed = load_model(checkpoint, torch.device('cpu'))
            self.assertFalse(removed)
            self.assertEqual(config, restored_config)
            restored.eval()
            with torch.no_grad():
                expected = model(dataset[0]['images'].unsqueeze(0))
                actual = restored(dataset[0]['images'].unsqueeze(0))
            for key in expected:
                torch.testing.assert_close(expected[key], actual[key])
            optimizer2 = torch.optim.AdamW(restored.parameters())
            scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, 1)
            optimizer2.load_state_dict(payload['optimizer'])
            scheduler2.load_state_dict(payload['scheduler'])
            self.assertEqual(optimizer.param_groups[0]['lr'], optimizer2.param_groups[0]['lr'])
            run_epoch(restored, loader, torch.device('cpu'), config.train, optimizer2)

    def test_legacy_weights_strictness(self):
        model = TLCFormer()
        state = model.state_dict()
        state['head2.0.weight'] = torch.zeros(1)
        state['signs_decoder.mul_arcface.weight'] = torch.zeros(1)
        removed = load_weights(model, {'module.' + key: value for key, value in state.items()})
        self.assertEqual(len(removed), 2)
        del state['query_embed.weight']
        with self.assertRaises(RuntimeError):
            load_weights(model, state)

    def test_config(self):
        config = load_config('configs/tlcformer.yaml')
        self.assertEqual(Config.from_dict(config.to_dict()), config)
        with self.assertRaises(ValueError):
            Config.from_dict({'model': {'num_levels': 2}})
        with self.assertRaises(ValueError):
            ModelConfig(num_heads=3)


if __name__ == '__main__':
    unittest.main()

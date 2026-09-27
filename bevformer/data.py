import glob
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import torchvision.transforms as T
import os
from pyquaternion import Quaternion
import yaml
import torchvision
import numpy as np
from collections import defaultdict
import re


normalize_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
))
def pad_or_trim_to_np(x, shape, pad_val=0):
  shape = np.asarray(shape)
  pad = shape - np.minimum(np.shape(x), shape)
  zeros = np.zeros_like(pad)
  x = np.pad(x, np.stack([zeros, pad], axis=1), constant_values=pad_val)
  return x[:shape[0], :shape[1]]

def img_transform(img, resize, resize_dims):
    post_rot2 = torch.eye(2)
    post_tran2 = torch.zeros(2)

    img = cv2.resize(img, resize_dims)

    rot_resize = torch.Tensor([[resize[0], 0],
                               [0, resize[1]]])
    post_rot2 = rot_resize @ post_rot2
    post_tran2 = rot_resize @ post_tran2

    post_tran = torch.zeros(3)
    post_rot = torch.eye(3)
    post_tran[:2] = post_tran2
    post_rot[:2, :2] = post_rot2
    return img, post_rot, post_tran

def drop_first_n_per_scene(path_list, n=5):

    scene_dict = defaultdict(list)

    for p in path_list:


        scene_dir = os.path.dirname(os.path.dirname(p))
        scene_dict[scene_dir].append(p)

    filtered = []
    for scene_dir in sorted(scene_dict.keys()):
        imgs = sorted(scene_dict[scene_dir])
        filtered.extend(imgs[n:])

    return filtered

TOWNS = ["Town01", "Town02", "Town03", "Town04", "Town05"]
TOWN_TO_IDX = {name: i for i, name in enumerate(TOWNS)}

def parse_town_from_path(path):

    m = re.search(r"(Town\d+)", path)
    if m is None:
        raise ValueError(f"Could not parse town from path: {path}")
    town_name = m.group(1)
    if town_name not in TOWN_TO_IDX:
        raise ValueError(f"Unknown town: {town_name}")
    return town_name, TOWN_TO_IDX[town_name]
class Drive_Dataset(Dataset):
    def __init__(self, train: bool,conf, v=1 ):
        self.v=v
        self.train=train

        if train:
            self.image_front = sorted(
                glob.glob('data/train/Town01/*/front_cam_rgb/*') +
                glob.glob('data/train/Town02/*/front_cam_rgb/*') +
                glob.glob('data/train/Town03/*/front_cam_rgb/*') +
                glob.glob('data/train/Town04/*/front_cam_rgb/*') +
                glob.glob('data/train/Town05/*/front_cam_rgb/*')
            )
        else:
            self.image_front = sorted(
                glob.glob('data/val/Town01/*/front_cam_rgb/*') +
                glob.glob('data/val/Town02/*/front_cam_rgb/*') +
                glob.glob('data/val/Town03/*/front_cam_rgb/*') +
                glob.glob('data/val/Town04/*/front_cam_rgb/*') +
                glob.glob('data/val/Town05/*/front_cam_rgb/*')
            )
        self.image_front = drop_first_n_per_scene(self.image_front, n=5)
        with open('data/sensor_config.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
        self.sensor_config = cfg['sensors']
        self.sensor_data = {
            'width': int(conf['width']),
            'height': int(conf['height']),
        }
        self.to_tensor = T.ToTensor()
    def __len__(self):
        return len(self.image_front)
    def get_cam_para(self):
        def get_cam_to_ego(dof):
            yaw = np.deg2rad(dof[5])

            rotation = Quaternion(
                scalar=np.cos(yaw / 2),
                vector=[0, 0, np.sin(yaw / 2)]
            )
            rotation_matrix = rotation.rotation_matrix

            adapter_matrix = np.array([
                [0, 0, 1],
                [1, 0, 0],
                [0, -1, 0]
            ], dtype=np.float32)

            final_rotation = rotation_matrix @ adapter_matrix


            translation = np.array(dof[:3], dtype=np.float32)[:, None]

            cam_to_ego = np.vstack([
                np.hstack((final_rotation, translation)),
                np.array([0, 0, 0, 1], dtype=np.float32)
            ]).astype(np.float32)

            return cam_to_ego, final_rotation, translation

        cam_names = [
            "right_cam",
            "front_cam",
            "left_cam",
            "rear_right_cam",
            "rear_cam",
            "rear_left_cam",
        ]

        extrinsic_list = []
        rotation_list = []
        translation_list = []
        intrinsic_list = []

        for cam in cam_names:
            cam_dof = [
                self.sensor_config[cam]['x'],
                self.sensor_config[cam]['y'],
                self.sensor_config[cam]['z'],
                self.sensor_config[cam]['pitch'],
                self.sensor_config[cam]['roll'],
                self.sensor_config[cam]['yaw'],
            ]

            cam_to_ego, final_rotation, translation = get_cam_to_ego(cam_dof)

            extrinsic_list.append(torch.from_numpy(cam_to_ego).float())
            rotation_list.append(torch.from_numpy(final_rotation).float())
            translation_list.append(torch.from_numpy(translation).float().squeeze(-1))
            orig_w = self.sensor_config[cam]['width']
            orig_h = self.sensor_config[cam]['height']
            out_w = self.sensor_data['width']
            out_h = self.sensor_data['height']
            fov = self.sensor_config[cam]['fov']

            sx = out_w / orig_w
            sy = out_h / orig_h

            f = orig_w / (2 * np.tan(np.deg2rad(fov) / 2))

            fx = f * sx
            fy = f * sy
            Cu = (orig_w / 2) * sx
            Cv = (orig_h / 2) * sy

            intrinsic = torch.tensor([
                [fx, 0,  Cu],
                [0,  fy, Cv],
                [0,  0,  1],
            ], dtype=torch.float32)
            intrinsic_list.append(intrinsic)

        extrinsic = torch.stack(extrinsic_list, dim=0)
        intrinsic = torch.stack(intrinsic_list, dim=0)
        rotation = torch.stack(rotation_list, dim=0)
        translation = torch.stack(translation_list, dim=0)

        return extrinsic, intrinsic, rotation, translation


    def sample_augmentation(self):
        fH, fW = self.sensor_data['height'] , self.sensor_data['width']
        resize = (fW / 400, fH / 300)
        resize_dims = (fW, fH)
        return resize, resize_dims

    def get_img(self, img_path):
        front_img_path = img_path.replace("front_cam_rgb", "front_cam_rgb")
        image_left_path = img_path.replace("front_cam_rgb", "left_cam_rgb")
        image_right_path = img_path.replace("front_cam_rgb", "right_cam_rgb")
        image_rear_path = img_path.replace("front_cam_rgb", "rear_cam_rgb")
        image_rear_left_path = img_path.replace("front_cam_rgb", "rear_left_cam_rgb")
        image_rear_right_path = img_path.replace("front_cam_rgb", "rear_right_cam_rgb")
        resize, resize_dims = self.sample_augmentation()


        if self.train:
            brightness = np.random.uniform(0.7, 1.3)
            contrast = np.random.uniform(0.7, 1.3)
            saturation = np.random.uniform(0.7, 1.3)
            hue = np.random.uniform(-0.1, 0.1)

        imgs = []
        post_rots = []
        post_trans = []
        for path in [image_right_path, front_img_path, image_left_path,
                    image_rear_right_path, image_rear_path, image_rear_left_path]:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img, post_rot, post_tran = img_transform(img, resize, resize_dims)

            if self.train:
                img_pil = T.ToPILImage()(img.astype(np.uint8))
                img_pil = T.functional.adjust_brightness(img_pil, brightness)
                img_pil = T.functional.adjust_contrast(img_pil, contrast)
                img_pil = T.functional.adjust_saturation(img_pil, saturation)
                img_pil = T.functional.adjust_hue(img_pil, hue)
                img = np.array(img_pil)

            img = normalize_img(img)
            imgs.append(img)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        imgs = torch.stack(imgs)
        return imgs, post_rots, post_trans

    def get_bev(self, img_path):

        das_mask = cv2.imread(img_path.replace("front_cam_rgb", "das"), cv2.IMREAD_GRAYSCALE)
        vehicle_mask = cv2.imread(img_path.replace('front_cam_rgb', 'vehicle_mask'), cv2.IMREAD_GRAYSCALE)
        stopline_mask = cv2.imread(img_path.replace('front_cam_rgb', 'stopline'), cv2.IMREAD_GRAYSCALE)
        walker_mask = cv2.imread(img_path.replace('front_cam_rgb', 'walker_mask'), cv2.IMREAD_GRAYSCALE)
        lane_broken_mask = cv2.imread(img_path.replace('front_cam_rgb', 'lane_broken'), cv2.IMREAD_GRAYSCALE)
        lane_solid_mask = cv2.imread(img_path.replace('front_cam_rgb', 'solid'), cv2.IMREAD_GRAYSCALE)


        das_mask = (das_mask == 255).astype('uint8') * 255
        stopline_mask = (stopline_mask == 255).astype('uint8') * 255
        vehicle_mask = (vehicle_mask == 255).astype('uint8') * 255
        walker_mask = (walker_mask == 255).astype('uint8') * 255
        lane_broken_mask = (lane_broken_mask == 255).astype('uint8') * 255
        lane_solid_mask = (lane_solid_mask == 255).astype('uint8') * 255

        das_mask = self.to_tensor(das_mask)
        lane_broken_mask = self.to_tensor(lane_broken_mask)
        lane_solid_mask = self.to_tensor(lane_solid_mask)
        vehicle_mask = self.to_tensor(vehicle_mask)
        walker_mask = self.to_tensor(walker_mask)
        stopline_mask = self.to_tensor(stopline_mask)

        return das_mask, lane_broken_mask, lane_solid_mask, vehicle_mask, walker_mask, stopline_mask

    def get_lidar(self,img_path):
        path = img_path.replace("front_cam_rgb", "lidar").replace(".png", ".npy")


        lidar_data = np.load(path, allow_pickle=True)

        points = lidar_data[:, :3]/2


        num_points = points.shape[0]
        lidar_data = pad_or_trim_to_np(points, [81920, 5]).astype('float32')
        lidar_mask = np.ones(81920).astype('float32')
        lidar_mask[num_points:] *= 0.0

        res=0.2
        x_range=(-20, 20)
        y_range=(-20, 20)
        image_size=(200, 200)

        x, y, _ = points[:, 0], points[:, 1], points[:, 2]


        x_rot = y
        y_rot = x


        x_img = ((x_rot - x_range[0]) / res).astype(np.int32)
        y_img = ((y_rot - y_range[0]) / res).astype(np.int32)


        x_img = np.clip(x_img, 0, image_size[0] - 1)
        y_img = np.clip(y_img, 0, image_size[1] - 1)


        topview_img = np.zeros(image_size, dtype=np.uint8)
        topview_img[image_size[1] - 1 - y_img, x_img] = 255


        return lidar_data, lidar_mask,self.to_tensor(topview_img)


    def __getitem__(self, idx):
        path = self.image_front[idx]

        bev_imgs, post_rots, post_trans = self.get_img(path)

        extrinsic, intrinsic, rotation, translation = self.get_cam_para()

        cam_pam = {
            'intrins': intrinsic,
            'rots': rotation,
            'trans': translation,
            'post_rots': post_rots,
            'post_trans': post_trans,
        }

        das, lane_broken,lane_solid, vehicle, walker, stopline_mask = self.get_bev(path)

        town_name, town_idx = parse_town_from_path(path)

        return (
            bev_imgs,
            cam_pam,
            das,
            lane_broken,lane_solid,
            vehicle,
            walker,
            stopline_mask,
            torch.tensor(town_idx, dtype=torch.long),
            town_name,
        )

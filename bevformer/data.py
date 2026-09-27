import glob
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import torchvision.transforms as T
import json
import os
from pyquaternion import Quaternion
import yaml
import torchvision
import numpy as np
import math
import random

def bezier_cubic(P0, P1, P2, P3, t):

    t = t[:, None]
    B = (1-t)**3*P0 + 3*(1-t)**2*t*P1 + 3*(1-t)*t**2*P2 + t**3*P3
    return B

def path_curvature_xy(path_xy, eps=1e-6):

    d1 = np.gradient(path_xy, axis=0)
    d2 = np.gradient(d1, axis=0)
    num = d1[:,0]*d2[:,1] - d1[:,1]*d2[:,0]
    den = (d1[:,0]**2 + d1[:,1]**2 + eps)**1.5
    return np.abs(num / (den + eps))

def line_collision_free(p0, p1, occ, step=1.0):
    if occ is None: return True
    H, W = occ.shape
    p0 = np.asarray(p0, np.float32)
    p1 = np.asarray(p1, np.float32)
    d = p1 - p0
    L = float(np.linalg.norm(d)) + 1e-6
    n = max(2, int(np.ceil(L/step)))
    for a in np.linspace(0, 1, n):
        x, y = p0 + a*d
        xi = int(round(np.clip(x, 0, W-1)))
        yi = int(round(np.clip(y, 0, H-1)))
        if occ[yi, xi] != 0:
            return False
    return True

def xy_to_xytheta(pts: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    T = pts.shape[0]
    if T == 1:
        return np.concatenate([pts, np.zeros((1,1), np.float32)], axis=1)
    d = np.zeros_like(pts)
    d[1:-1] = pts[2:] - pts[:-2]
    d[0]    = pts[1]  - pts[0]
    d[-1]   = pts[-1] - pts[-2]
    theta = np.arctan2(d[:,1] + eps, d[:,0] + eps)
    theta_unwrapped = np.unwrap(theta.astype(np.float64)).astype(np.float32)
    return np.concatenate([pts, theta_unwrapped[:,None]], axis=1)


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
def apply_color_aug_rgb(img_rgb, p):

    img = img_rgb.astype(np.float32)


    mean = img.mean(axis=(0, 1), keepdims=True)
    img = (img - mean) * p["contrast"] + mean
    img = img * p["brightness"]
    img = np.clip(img, 0, 255)


    img[..., 0] *= p["r_gain"]
    img[..., 1] *= p["g_gain"]
    img[..., 2] *= p["b_gain"]
    img = np.clip(img, 0, 255).astype(np.uint8)


    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)

    hsv[..., 0] = (hsv[..., 0] + p["hue_shift"]) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * p["sat_scale"], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * p["val_scale"], 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


    if p["do_clahe"]:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


    gamma = p["gamma"]
    if abs(gamma - 1.0) > 1e-3:
        table = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)]).astype("uint8")
        img = cv2.LUT(img, table)

    return img

class Drive_Dataset(Dataset):
    def __init__(self, train: bool,conf, seq_len=0, stride=1, ):
        self.seq_len = seq_len
        self.stride = stride
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
            yaw = dof[5]
            rotation = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)])
            rotation_matrix = rotation.rotation_matrix

            adapter_matrix = np.array([
                [0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0]
            ])

            final_rotation = rotation_matrix @ adapter_matrix

            translation = np.array(dof[:3])[:, None]

            cam_to_ego = np.vstack([
                np.hstack((final_rotation, translation)),
                np.array([0, 0, 0, 1])
            ])

            return cam_to_ego, final_rotation, translation

        cam_names = ['left_cam', 'front_cam', 'right_cam', 'rear_left_cam','rear_cam', 'rear_right_cam']

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


            extrinsic_list.append(torch.from_numpy(cam_to_ego).float().unsqueeze(0))
            rotation_list.append(torch.from_numpy(final_rotation).float().unsqueeze(0))
            translation_list.append(torch.from_numpy(translation).float().unsqueeze(0))

            w = self.sensor_config[cam]['width']
            h = self.sensor_config[cam]['height']
            fov = self.sensor_config[cam]['fov']
            f = w / (2 * np.tan(fov * np.pi / 360))
            Cu = w / 2
            Cv = h / 2
            intrinsic = torch.tensor([
                [f, 0, Cu],
                [0, f, Cv],
                [0, 0, 1]
            ], dtype=torch.float32).unsqueeze(0)
            intrinsic_list.append(intrinsic)

        extrinsic = torch.cat(extrinsic_list, dim=0)
        rotation = torch.cat(rotation_list, dim=0)
        translation = torch.cat(translation_list, dim=0).squeeze(-1)
        intrinsic = torch.cat(intrinsic_list, dim=0)

        return extrinsic, intrinsic, rotation, translation


    def sample_augmentation(self):
        fH, fW = self.sensor_data['height'] , self.sensor_data['width']
        resize = (fW / 400, fH / 300)
        resize_dims = (fW, fH)
        return resize, resize_dims

    def sample_color_aug_params(self):
        params = {
            "brightness": random.uniform(0.85, 1.20),
            "contrast":   random.uniform(0.85, 1.25),

            "sat_scale":  random.uniform(0.80, 1.50),
            "hue_shift":  random.uniform(-8, 8),
            "val_scale":  random.uniform(0.90, 1.15),

            "gamma":      random.uniform(0.85, 1.20),

            "r_gain":     random.uniform(0.90, 1.10),
            "g_gain":     random.uniform(0.90, 1.10),
            "b_gain":     random.uniform(0.90, 1.10),

            "do_clahe":   random.random() < 0.2,
        }
        return params


    def get_img(self, img_path):

        front_img_path = img_path.replace("front_cam_rgb", "front_cam_rgb")
        image_left_path = img_path.replace("front_cam_rgb", "left_cam_rgb")
        image_right_path = img_path.replace("front_cam_rgb", "right_cam_rgb")

        image_rear_path = img_path.replace("front_cam_rgb", "rear_cam_rgb")
        image_rear_left_path = img_path.replace("front_cam_rgb", "rear_left_cam_rgb")
        image_rear_right_path = img_path.replace("front_cam_rgb", "rear_right_cam_rgb")
        resize, resize_dims = self.sample_augmentation()
        color_aug_params = self.sample_color_aug_params()

        imgs = []
        post_rots = []
        post_trans = []
        for path in [image_left_path, front_img_path, image_right_path, image_rear_left_path, image_rear_path,image_rear_right_path]:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img, post_rot, post_tran = img_transform(img, resize, resize_dims)
            if self.train:
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                img = apply_color_aug_rgb(img, color_aug_params)

            img = normalize_img(img)

            imgs.append(img)
            post_rots.append(post_rot)
            post_trans.append(post_tran)


        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        imgs = torch.stack(imgs)
        return imgs, post_rots, post_trans

    def get_bev(self, img_path):
        bev_path=img_path.replace("front_cam_rgb", "das")

        top = cv2.imread(bev_path)
        das_mask = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)

        vehicle_mask = cv2.imread(bev_path.replace('das', 'vehicle_mask'), cv2.IMREAD_GRAYSCALE)
        route_mask = cv2.imread(bev_path.replace('das', 'road_network'), cv2.IMREAD_GRAYSCALE)
        stopline_mask = cv2.imread(bev_path.replace('das', 'stopline'), cv2.IMREAD_GRAYSCALE)
        walker_mask = cv2.imread(bev_path.replace('das', 'walker_mask'), cv2.IMREAD_GRAYSCALE)
        lane_mask = cv2.imread(bev_path.replace('das', 'lane'), cv2.IMREAD_GRAYSCALE)


        das_mask = (das_mask == 255).astype('uint8')*255
        stopline_mask = (stopline_mask == 255).astype('uint8')*255
        route_mask = (route_mask == 255).astype('uint8')*255
        vehicle_mask = (vehicle_mask == 255).astype('uint8')*255
        walker_mask = (walker_mask == 255).astype('uint8')*255
        lane_mask = (lane_mask == 255).astype('uint8')*255


        vehicle_mask = vehicle_mask[150:350, 150:350]
        walker_mask = walker_mask[150:350, 150:350]


        das_mask = self.to_tensor(das_mask)
        lane_mask = self.to_tensor(lane_mask)
        vehicle_mask = self.to_tensor(vehicle_mask)
        walker_mask = self.to_tensor(walker_mask)

        stopline_mask = self.to_tensor(stopline_mask)
        route_mask = self.to_tensor(route_mask)

        return das_mask, lane_mask, vehicle_mask, walker_mask,route_mask,stopline_mask


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
        path=self.image_front[idx]

        bev_imgs, post_rots, post_trans = self.get_img(path)

        lidar_data, lidar_mask,topview_img=self.get_lidar(path)
        extrinsic, intrinsic, rotation, translation = self.get_cam_para()
        cam_pam={
            'intrins':intrinsic,
            'rots':rotation,
            'trans':translation,
            'post_rots':post_rots,
            'post_trans':post_trans,

            }

        das, lane, vehicle, walker, route_mask, stopline_mask = self.get_bev(path)


        return bev_imgs,topview_img,cam_pam, lidar_data, lidar_mask,das, lane, vehicle, walker, stopline_mask,route_mask

import os
import random

import imageio
import matplotlib
import matplotlib.colors
import numpy as np
import skimage.transform
import torch
import torchvision
from torch.utils.data import Dataset

image_h = 384
image_w = 768


class FreiburgForest(Dataset):
    def __init__(self, transform=None, data_dir=None, fraction=None):
        self.transform = transform

        self.rgb_images = []
        self.evi2_images = []
        self.gt_images = []
        self.basenames = []

        rgb_folder_name = 'rgb'
        evi2_gray_folder_name = 'evi2_gray'
        gt_folder_name = 'GT'

        for filename_w_ext in os.listdir(os.path.join(data_dir, rgb_folder_name)):
            filename, _ = os.path.splitext(filename_w_ext)
            basename = filename.split('_')[0]

            rgb_path = os.path.join(os.path.join(data_dir, rgb_folder_name), f'{basename}_Clipped.jpg')
            evi2_gray_path = os.path.join(os.path.join(data_dir, evi2_gray_folder_name), f'{basename}.tif')
            gt_path = os.path.join(os.path.join(data_dir, gt_folder_name), f'{basename}_mask.png')

            self.rgb_images.append(imageio.imread(rgb_path))
            self.evi2_images.append(imageio.imread(evi2_gray_path))
            self.gt_images.append(imageio.imread(gt_path).astype(float))  # needed for skimage.transform.resize
            self.basenames.append(basename)

        if fraction is not None:  # Only use a fraction of the data in the directory
            len_data = len(self.rgb_images)
            random_indices = random.sample(list(range(len_data)), int(fraction * len_data))
            self.rgb_images = [self.rgb_images[index] for index in random_indices]
            self.evi2_images = [self.evi2_images[index] for index in random_indices]
            self.gt_images = [self.gt_images[index] for index in random_indices]
            self.basenames = [self.basenames[index] for index in random_indices]

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        sample = {'rgb': self.rgb_images[idx], 'evi': self.evi2_images[idx], 'label': self.gt_images[idx]}

        if self.transform:
            sample = self.transform(sample)

        sample['basename'] = self.basenames[idx]

        return sample


class RandomHSV(object):
    """
    Args:
        h_range (float tuple): random ratio of the hue channel,
            new_h range from h_range[0]*old_h to h_range[1]*old_h.
        s_range (float tuple): random ratio of the saturation channel,
            new_s range from s_range[0]*old_s to s_range[1]*old_s.
        v_range (int tuple): random bias of the value channel,
            new_v range from old_v-v_range to old_v+v_range.
    Notice:
        h range: 0-1
        s range: 0-1
        v range: 0-255
    """

    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, sample):
        img = sample['rgb']
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        return {'rgb': img_new, 'evi': sample['evi'], 'label': sample['label']}


class ScaleNorm(object):
    def __call__(self, sample):
        rgb, evi, label = sample['rgb'], sample['evi'], sample['label']

        # Bi-linear
        rgb = skimage.transform.resize(rgb, (image_h, image_w), order=1,
                                       mode='reflect', preserve_range=True)
        # Nearest-neighbor
        evi = skimage.transform.resize(evi, (image_h, image_w), order=0,
                                       mode='reflect', preserve_range=True)
        label = skimage.transform.resize(label, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)

        return {'rgb': rgb, 'evi': evi, 'label': label}


class RandomScale(object):
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        rgb, evi, label = sample['rgb'], sample['evi'], sample['label']

        target_scale = random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * rgb.shape[0]))
        target_width = int(round(target_scale * rgb.shape[1]))
        # Bi-linear
        rgb = skimage.transform.resize(rgb, (target_height, target_width),
                                       order=1, mode='reflect', preserve_range=True)
        # Nearest-neighbor
        evi = skimage.transform.resize(evi, (target_height, target_width),
                                       order=0, mode='reflect', preserve_range=True)
        label = skimage.transform.resize(label, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)

        return {'rgb': rgb, 'evi': evi, 'label': label}


class RandomCrop(object):
    def __init__(self, th, tw):
        self.th = th
        self.tw = tw

    def __call__(self, sample):
        rgb, evi, label = sample['rgb'], sample['evi'], sample['label']
        h = rgb.shape[0]
        w = rgb.shape[1]
        i = random.randint(0, h - self.th)
        j = random.randint(0, w - self.tw)

        return {'rgb': rgb[i:i + image_h, j:j + image_w, :],
                'evi': evi[i:i + image_h, j:j + image_w],
                'label': label[i:i + image_h, j:j + image_w]}


class RandomFlip(object):
    def __call__(self, sample):
        rgb, evi, label = sample['rgb'], sample['evi'], sample['label']
        if random.random() > 0.5:
            rgb = np.fliplr(rgb).copy()
            evi = np.fliplr(evi).copy()
            label = np.fliplr(label).copy()

        return {'rgb': rgb, 'evi': evi, 'label': label}


class RandomRotate(object):
    def __init__(self, angle_range):
        self.angle_low = min(angle_range)
        self.angle_high = max(angle_range)

    def __call__(self, sample):
        rgb, evi, label = sample['rgb'], sample['evi'], sample['label']

        angle = random.uniform(self.angle_low, self.angle_high)

        rgb = skimage.transform.rotate(rgb, angle=angle, order=1, mode='reflect', preserve_range=True)
        evi = skimage.transform.rotate(evi, angle=angle, order=0, mode='reflect', preserve_range=True)
        label = skimage.transform.rotate(label, angle=angle, order=0, mode='reflect', preserve_range=True)

        return {'rgb': rgb, 'evi': evi, 'label': label}


class RandomSkew(object):
    def __init__(self, skew_range):
        self.skew_low = min(skew_range)
        self.skew_high = max(skew_range)

    def __call__(self, sample):
        rgb, evi, label = sample['rgb'], sample['evi'], sample['label']

        affine_tf = skimage.transform.AffineTransform(shear=random.uniform(self.skew_low, self.skew_high))

        rgb = skimage.transform.warp(rgb, inverse_map=affine_tf, order=1, mode='reflect', preserve_range=True)
        evi = skimage.transform.warp(evi, inverse_map=affine_tf, order=0, mode='reflect', preserve_range=True)
        label = skimage.transform.warp(label, inverse_map=affine_tf, order=0, mode='reflect', preserve_range=True)

        return {'rgb': rgb, 'evi': evi, 'label': label}


# Transforms on torch.*Tensor
class ToZeroOneRange(object):
    def __call__(self, sample):
        rgb, evi = sample['rgb'], sample['evi']
        rgb = rgb / 255.
        evi = evi / 255.

        sample['rgb'] = rgb
        sample['evi'] = evi

        return sample


# Transforms on torch.*Tensor
class Normalize(object):
    def __call__(self, sample):
        rgb, evi = sample['rgb'], sample['evi']
        rgb = rgb / 255.
        evi = evi / 255.

        rgb = torchvision.transforms.Normalize(mean=[0.462971, 0.397023, 0.326541],
                                               std=[0.293528, 0.293154, 0.307788])(rgb)
        evi = torchvision.transforms.Normalize(mean=[0.632958],
                                               std=[0.150213])(evi)

        sample['rgb'] = rgb
        sample['evi'] = evi

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        rgb, evi, label = sample['rgb'], sample['evi'], sample['label']

        # Generate different label scales
        label2 = skimage.transform.resize(label, (label.shape[0] // 2, label.shape[1] // 2),
                                          order=0, mode='reflect', preserve_range=True)
        label3 = skimage.transform.resize(label, (label.shape[0] // 4, label.shape[1] // 4),
                                          order=0, mode='reflect', preserve_range=True)
        label4 = skimage.transform.resize(label, (label.shape[0] // 8, label.shape[1] // 8),
                                          order=0, mode='reflect', preserve_range=True)
        label5 = skimage.transform.resize(label, (label.shape[0] // 16, label.shape[1] // 16),
                                          order=0, mode='reflect', preserve_range=True)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        rgb = rgb.transpose((2, 0, 1))
        evi = np.expand_dims(evi, 0).astype(np.float)
        return {'rgb': torch.from_numpy(rgb).float(),
                'evi': torch.from_numpy(evi).float(),
                'label': torch.from_numpy(label).float(),
                'label2': torch.from_numpy(label2).float(),
                'label3': torch.from_numpy(label3).float(),
                'label4': torch.from_numpy(label4).float(),
                'label5': torch.from_numpy(label5).float()}

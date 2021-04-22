import glob
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
    def __init__(self, transform=None, data_dirs=None, modal1_name='rgb', modal2_name='evi2_gray',
                 gt_available=True, fraction=None):
        self.transform = transform

        self.modal1_images = []
        self.modal2_images = []
        self.gt_images = []
        self.basenames = []

        modal1_folder_name = modal1_name
        modal2_folder_name = modal2_name
        gt_folder_name = 'GT'

        for data_dir in data_dirs:
            for filename_w_ext in os.listdir(os.path.join(data_dir, modal1_folder_name)):
                filename, _ = os.path.splitext(filename_w_ext)
                basename = filename.split('_')[0]

                modal1_path = glob.glob(os.path.join(os.path.join(data_dir, modal1_folder_name), f'{basename}*'))[0]
                modal2_path = glob.glob(os.path.join(os.path.join(data_dir, modal2_folder_name), f'{basename}*'))[0]

                self.modal1_images.append(imageio.imread(modal1_path))
                self.modal2_images.append(imageio.imread(modal2_path))
                self.basenames.append(basename)

                if gt_available:
                    gt_path = glob.glob(os.path.join(os.path.join(data_dir, gt_folder_name), f'{basename}*'))[0]
                    self.gt_images.append(imageio.imread(gt_path).astype(float))  # needed for skimage.transform.resize
                else:
                    self.gt_images.append(np.zeros_like(imageio.imread(modal2_path)).astype(float))

        if fraction is not None:  # Only use a fraction of the data in the directory
            len_data = len(self.modal1_images)
            random_indices = random.sample(list(range(len_data)), int(fraction * len_data))
            self.modal1_images = [self.modal1_images[index] for index in random_indices]
            self.modal2_images = [self.modal2_images[index] for index in random_indices]
            self.gt_images = [self.gt_images[index] for index in random_indices]
            self.basenames = [self.basenames[index] for index in random_indices]

    def __len__(self):
        return len(self.modal1_images)

    def __getitem__(self, idx):
        sample = {'modal1': self.modal1_images[idx], 'modal2': self.modal2_images[idx], 'label': self.gt_images[idx]}

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
        img = sample['modal1']
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

        return {'modal1': img_new, 'modal2': sample['modal2'], 'label': sample['label']}


class ScaleNorm(object):
    def __call__(self, sample):
        modal1, modal2, label = sample['modal1'], sample['modal2'], sample['label']

        # Bi-linear
        modal1 = skimage.transform.resize(modal1, (image_h, image_w), order=1,
                                          mode='reflect', preserve_range=True)
        # Nearest-neighbor
        modal2 = skimage.transform.resize(modal2, (image_h, image_w), order=0,
                                          mode='reflect', preserve_range=True)
        label = skimage.transform.resize(label, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)

        return {'modal1': modal1, 'modal2': modal2, 'label': label}


class RandomScale(object):
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        modal1, modal2, label = sample['modal1'], sample['modal2'], sample['label']

        target_scale = random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * modal1.shape[0]))
        target_width = int(round(target_scale * modal1.shape[1]))
        # Bi-linear
        modal1 = skimage.transform.resize(modal1, (target_height, target_width),
                                          order=1, mode='reflect', preserve_range=True)
        # Nearest-neighbor
        modal2 = skimage.transform.resize(modal2, (target_height, target_width),
                                          order=0, mode='reflect', preserve_range=True)
        label = skimage.transform.resize(label, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)

        return {'modal1': modal1, 'modal2': modal2, 'label': label}


class RandomCrop(object):
    def __init__(self, th, tw):
        self.th = th
        self.tw = tw

    def __call__(self, sample):
        modal1, modal2, label = sample['modal1'], sample['modal2'], sample['label']
        h = modal1.shape[0]
        w = modal1.shape[1]
        i = random.randint(0, h - self.th)
        j = random.randint(0, w - self.tw)

        return {'modal1': modal1[i:i + image_h, j:j + image_w, :],
                'modal2': modal2[i:i + image_h, j:j + image_w],
                'label': label[i:i + image_h, j:j + image_w]}


class RandomFlip(object):
    def __call__(self, sample):
        modal1, modal2, label = sample['modal1'], sample['modal2'], sample['label']
        if random.random() > 0.5:
            modal1 = np.fliplr(modal1).copy()
            modal2 = np.fliplr(modal2).copy()
            label = np.fliplr(label).copy()

        return {'modal1': modal1, 'modal2': modal2, 'label': label}


class RandomRotate(object):
    def __init__(self, angle_range):
        self.angle_low = min(angle_range)
        self.angle_high = max(angle_range)

    def __call__(self, sample):
        modal1, modal2, label = sample['modal1'], sample['modal2'], sample['label']

        angle = random.uniform(self.angle_low, self.angle_high)

        modal1 = skimage.transform.rotate(modal1, angle=angle, order=1, mode='reflect', preserve_range=True)
        modal2 = skimage.transform.rotate(modal2, angle=angle, order=0, mode='reflect', preserve_range=True)
        label = skimage.transform.rotate(label, angle=angle, order=0, mode='reflect', preserve_range=True)

        return {'modal1': modal1, 'modal2': modal2, 'label': label}


class RandomSkew(object):
    def __init__(self, skew_range):
        self.skew_low = min(skew_range)
        self.skew_high = max(skew_range)

    def __call__(self, sample):
        modal1, modal2, label = sample['modal1'], sample['modal2'], sample['label']

        affine_tf = skimage.transform.AffineTransform(shear=random.uniform(self.skew_low, self.skew_high))

        modal1 = skimage.transform.warp(modal1, inverse_map=affine_tf, order=1, mode='reflect', preserve_range=True)
        modal2 = skimage.transform.warp(modal2, inverse_map=affine_tf, order=0, mode='reflect', preserve_range=True)
        label = skimage.transform.warp(label, inverse_map=affine_tf, order=0, mode='reflect', preserve_range=True)

        return {'modal1': modal1, 'modal2': modal2, 'label': label}


# Transforms on torch.*Tensor
class ToZeroOneRange(object):
    def __call__(self, sample):
        modal1, modal2 = sample['modal1'], sample['modal2']
        modal1 = modal1 / 255.
        modal2 = modal2 / 255.

        sample['modal1'] = modal1
        sample['modal2'] = modal2

        return sample


# Transforms on torch.*Tensor
class Normalize(object):
    def __init__(self, modal1_name='rgb', modal2_name='evi2_gray'):
        self.modal1_name = modal1_name
        self.modal2_name = modal2_name

    def __call__(self, sample):
        modal1, modal2 = sample['modal1'], sample['modal2']
        modal1 = modal1 / 255.
        modal2 = modal2 / 255.

        mean_modality = {
            'rgb': [0.462971, 0.397023, 0.326541],
            'evi2_gray': [0.632958],
        }

        std_modality = {
            'rgb': [0.293528, 0.293154, 0.307788],
            'evi2_gray': [0.150213],
        }

        modal1 = torchvision.transforms.Normalize(mean=mean_modality[self.modal1_name],
                                                  std=std_modality[self.modal1_name])(modal1)
        modal2 = torchvision.transforms.Normalize(mean=mean_modality[self.modal2_name],
                                                  std=std_modality[self.modal2_name])(modal2)

        sample['modal1'] = modal1
        sample['modal2'] = modal2

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        modal1, modal2, label = sample['modal1'], sample['modal2'], sample['label']

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
        modal1 = modal1.transpose((2, 0, 1))
        modal2 = np.expand_dims(modal2, 0).astype(np.float)
        return {'modal1': torch.from_numpy(modal1).float(),
                'modal2': torch.from_numpy(modal2).float(),
                'label': torch.from_numpy(label).float(),
                'label2': torch.from_numpy(label2).float(),
                'label3': torch.from_numpy(label3).float(),
                'label4': torch.from_numpy(label4).float(),
                'label5': torch.from_numpy(label5).float()}

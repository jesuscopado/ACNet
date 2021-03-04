import numpy as np
import scipy.io
import imageio
import h5py
import os
from torch.utils.data import Dataset
import matplotlib
import matplotlib.colors
import skimage.transform
import random
import torchvision
import torch

import imageio

image_h = 384
image_w = 768


class FreiburgForest(Dataset):
    def __init__(self, transform=None, data_dir=None):
        self.transform = transform

        self.rgb_images = []
        self.evi2_images = []
        self.gt_images = []
        self.image_names = []

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
            gt_test = imageio.imread(gt_path)
            if np.any(gt_test == 0):
                print(gt_path)
                raise ValueError
            self.gt_images.append(imageio.imread(gt_path))
            self.image_names.append(basename)

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        sample = {'image': self.rgb_images[idx], 'depth': self.evi2_images[idx], 'label': self.gt_images[idx]}

        if self.transform:
            sample = self.transform(sample)

        if np.any(sample['label'].clone().cpu().data.numpy() == 0):
            print('after augment break')
            print(np.where(sample['label'].clone().cpu().data.numpy() == 0))
            print(self.image_names[idx])
            raise ValueError

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
        img = sample['image']
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

        return {'image': img_new, 'depth': sample['depth'], 'label': sample['label']}


class scaleNorm(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        # Bi-linear
        image = skimage.transform.resize(image, (image_h, image_w), order=1,
                                         mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)
        label = skimage.transform.resize(label, (image_h, image_w), order=0,
                                         mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}


class RandomScale(object):
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

        target_scale = random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * image.shape[0]))
        target_width = int(round(target_scale * image.shape[1]))
        # Bi-linear
        image = skimage.transform.resize(image, (target_height, target_width),
                                         order=1, mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = skimage.transform.resize(depth, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)
        label = skimage.transform.resize(label, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)

        return {'image': image, 'depth': depth, 'label': label}


class RandomCrop(object):
    def __init__(self, th, tw):
        self.th = th
        self.tw = tw

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        h = image.shape[0]
        w = image.shape[1]
        i = random.randint(0, h - self.th)
        j = random.randint(0, w - self.tw)

        return {'image': image[i:i + image_h, j:j + image_w, :],
                'depth': depth[i:i + image_h, j:j + image_w],
                'label': label[i:i + image_h, j:j + image_w]}


class RandomFlip(object):
    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()
            label = np.fliplr(label).copy()

        return {'image': image, 'depth': depth, 'label': label}


# Transforms on torch.*Tensor
class Normalize(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        image = image / 255.
        depth = depth / 255.

        '''
        # image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                          std=[0.229, 0.224, 0.225])(image)
        image = torchvision.transforms.Normalize(mean=[0.4850042694973687, 0.41627756261047333, 0.3981809741523051],
                                                 std=[0.26415541082494515, 0.2728415392982039, 0.2831175140191598])(
            image)
        depth = torchvision.transforms.Normalize(mean=[2.8424503515351494],
                                                 std=[0.9932836506164299])(depth)
        '''
        # TODO: compute mean and std for evi2_gray images

        sample['image'] = image
        sample['depth'] = depth

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, depth, label = sample['image'], sample['depth'], sample['label']

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
        image = image.transpose((2, 0, 1))
        depth = np.expand_dims(depth, 0).astype(np.float)
        return {'image': torch.from_numpy(image).float(),
                'depth': torch.from_numpy(depth).float(),
                'label': torch.from_numpy(label).float(),
                'label2': torch.from_numpy(label2).float(),
                'label3': torch.from_numpy(label3).float(),
                'label4': torch.from_numpy(label4).float(),
                'label5': torch.from_numpy(label5).float()}

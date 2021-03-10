import argparse
import os

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import ACNet_data_freiburgforest as ACNet_data

parser = argparse.ArgumentParser()
parser.add_argument('--train-dir', default=None, metavar='DIR',
                    help='path to labels')
parser.add_argument('--output-dir', default='data/',
                    help='path to dir where to write the result')
args = parser.parse_args()


def main():
    train_data = ACNet_data.FreiburgForest(
        transform=transforms.Compose([
            ACNet_data.ScaleNorm(),
            ACNet_data.Normalize()
        ]),
        data_dir=args.train_dir
    )
    loader = DataLoader(train_data, batch_size=len(train_data), num_workers=1)
    sample = next(iter(loader))
    rgb_mean = sample['rgb'].mean((0, 1, 2)).numpy()
    rgb_std = sample['rgb'].std((0, 1, 2)).numpy()
    evi_mean = sample['evi'].mean().unsqueeze(0).numpy()
    evi_std = sample['evi'].std().unsqueeze(0).numpy()
    np.savetxt(os.path.join(args.output_dir, 'rgb_mean.txt'), rgb_mean, delimiter='\n', fmt='%.6f')
    np.savetxt(os.path.join(args.output_dir, 'rgb_std.txt'), rgb_std, delimiter='\n', fmt='%.6f')
    np.savetxt(os.path.join(args.output_dir, 'evi_mean.txt'), evi_mean, delimiter='\n', fmt='%.6f')
    np.savetxt(os.path.join(args.output_dir, 'evi_std.txt'), evi_std, delimiter='\n', fmt='%.6f')


if __name__ == '__main__':
    main()

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
parser.add_argument('--modal1', default='rgb', help='Modality 1 for the model (3 channels)')
parser.add_argument('--modal2', default='evi2_gray', help='Modality 2 for the model (1 channel)')
args = parser.parse_args()


def main():
    train_data = ACNet_data.FreiburgForest(
        transform=transforms.Compose([
            ACNet_data.ScaleNorm(),
            ACNet_data.ToTensor(),
            ACNet_data.ToZeroOneRange()
        ]),
        data_dirs=[args.train_dir],
        modal1_name=args.modal1,
        modal2_name=args.modal2
    )
    loader = DataLoader(train_data, batch_size=len(train_data), num_workers=1)
    sample = next(iter(loader))
    rgb_mean = sample['modal1'].mean((0, 2, 3)).numpy()
    rgb_std = sample['modal1'].std((0, 2, 3)).numpy()
    evi_mean = sample['modal2'].mean().unsqueeze(0).numpy()
    evi_std = sample['modal2'].std().unsqueeze(0).numpy()
    np.savetxt(os.path.join(args.output_dir, 'rgb_mean.txt'), rgb_mean, delimiter='\n', fmt='%.6f')
    np.savetxt(os.path.join(args.output_dir, 'rgb_std.txt'), rgb_std, delimiter='\n', fmt='%.6f')
    np.savetxt(os.path.join(args.output_dir, 'evi_mean.txt'), evi_mean, delimiter='\n', fmt='%.6f')
    np.savetxt(os.path.join(args.output_dir, 'evi_std.txt'), evi_std, delimiter='\n', fmt='%.6f')


if __name__ == '__main__':
    main()

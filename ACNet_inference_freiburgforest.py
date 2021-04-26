import argparse
import os

import imageio
import numpy as np
import torch
import torch.optim
import torchvision
from torch.utils.data import DataLoader

import ACNet_data_freiburgforest as ACNet_data
# import ACNet_models
import ACNet_models_V1
# import ACNet_models_V1_first as ACNet_models_V1
# import ACNet_models_V1_delA as ACNet_models_V1
from utils import utils
from utils.utils import load_ckpt

parser = argparse.ArgumentParser(description='Multimodal Semantic Segmentation')
parser.add_argument('--data-dir', default=None, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--modal1', default='rgb', help='Modality 1 for the model (3 channels)')
parser.add_argument('--modal2', default='evi2', help='Modality 2 for the model (1 channel)')
parser.add_argument('-o', '--output-dir', default='./pred/', metavar='DIR',
                    help='path to output')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--num-class', default=5, type=int,
                    help='number of classes')
parser.add_argument('--visualize', default=False, action='store_true',
                    help='if output image')
parser.add_argument('--save-predictions', default=True, action='store_true',
                    help='if all predictions want to be saved')

args = parser.parse_args()
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")


def inference():
    model = ACNet_models_V1.ACNet(num_class=5, pretrained=False)
    load_ckpt(model, None, None, args.last_ckpt, device)
    model.eval()
    model.to(device)

    data = ACNet_data.FreiburgForest(
        transform=torchvision.transforms.Compose([
            ACNet_data.ScaleNorm(),
            ACNet_data.ToTensor(),
            ACNet_data.Normalize()
        ]),
        data_dirs=[args.data_dir],
        modal1_name=args.modal1,
        modal2_name=args.modal2,
        gt_available=False
    )
    data_loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    with torch.no_grad():
        for batch_idx, sample in enumerate(data_loader):
            modal1 = sample['modal1'].to(device)
            modal2 = sample['modal2'].to(device)
            basename = sample['basename'][0]

            with torch.no_grad():
                pred = model(modal1, modal2)

            output = torch.argmax(pred, 1) + 1
            output = output.squeeze(0).cpu().numpy()

            if args.save_predictions:
                colored_output = utils.color_label_eval(output).astype(np.uint8)
                imageio.imwrite(f'{args.output_dir}/{basename}_pred.png', colored_output.transpose([1, 2, 0]))


if __name__ == '__main__':
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    inference()

import argparse
import datetime
import os

import cv2
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
from utils.utils import load_ckpt, intersectionAndUnion, AverageMeter, accuracy, macc

parser = argparse.ArgumentParser(description='RGBD Sementic Segmentation')
parser.add_argument('--test-dir', default=None, metavar='DIR',
                    help='path to test dataset')
parser.add_argument('-o', '--output-dir', default='./result/', metavar='DIR',
                    help='path to output')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--num-class', default=5, type=int,
                    help='number of classes')
parser.add_argument('--visualize', default=False, action='store_true',
                    help='if output image')
parser.add_argument('--save-predictions', default=False, action='store_true',
                    help='if all predictions want to be saved')

args = parser.parse_args()
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")


def visualize_result(img, evi, label, preds, info, args):
    # segmentation
    img = img.squeeze(0).transpose(0, 2, 1)
    evi = evi.squeeze(0).squeeze(0)
    evi = (evi * 255 / evi.max()).astype(np.uint8)
    evi = cv2.applyColorMap(evi, cv2.COLORMAP_JET)
    evi = evi.transpose(2, 1, 0)
    seg_color = utils.color_label_eval(label)
    # prediction
    pred_color = utils.color_label_eval(preds)

    # aggregate images and save
    im_vis = np.concatenate((img, evi, seg_color, pred_color), axis=1).astype(np.uint8)
    im_vis = im_vis.transpose(2, 1, 0)

    img_name = str(info)
    # print('write check: ', im_vis.dtype)
    cv2.imwrite(os.path.join(args.output_dir, img_name + '.png'), im_vis)


def evaluate():
    model = ACNet_models_V1.ACNet(num_class=5, pretrained=False)
    load_ckpt(model, None, args.last_ckpt, device)
    model.eval()
    model.to(device)

    val_data = ACNet_data.FreiburgForest(
        transform=torchvision.transforms.Compose([
            ACNet_data.scaleNorm(),
            ACNet_data.ToTensor(),
            ACNet_data.Normalize()
        ]),
        data_dir=args.test_dir
    )
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    a_meter = AverageMeter()
    b_meter = AverageMeter()
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            rgb = sample['rgb'].to(device)
            evi = sample['evi'].to(device)
            label = sample['label'].numpy()
            basename = sample['basename'][0]

            with torch.no_grad():
                pred = model(rgb, evi)

            output = torch.argmax(pred, 1) + 1
            output = output.squeeze(0).cpu().numpy()

            acc, pix = accuracy(output, label)
            intersection, union = intersectionAndUnion(output, label, args.num_class)
            acc_meter.update(acc, pix)
            a_m, b_m = macc(output, label, args.num_class)
            intersection_meter.update(intersection)
            union_meter.update(union)
            a_meter.update(a_m)
            b_meter.update(b_m)
            print('[{}] iter {}, accuracy: {}'
                  .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), batch_idx, acc))

            if args.visualize:
                visualize_result(rgb, evi, label, output, batch_idx, args)

            if args.save_predictions:
                colored_output = utils.color_label_eval(output).astype(np.uint8)
                imageio.imwrite(f'{args.output_dir}/{basename}_pred.png', colored_output.transpose([1, 2, 0]))

    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {}'.format(i, _iou))

    mAcc = (a_meter.average() / (b_meter.average() + 1e-10))
    print(mAcc.mean())
    print('[Eval Summary]:')
    print('Mean IoU: {:.4}, Accuracy: {:.2f}%'
          .format(iou.mean(), acc_meter.average() * 100))


if __name__ == '__main__':
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    evaluate()

'''
Our code is partially adapted from RedNet (https://github.com/JinDongJiang/RedNet)
'''
import argparse
import os

import numpy as np
import torch
import torch.optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

import ACNet_data_freiburgforest as ACNet_data
import ACNet_models_V1
from utils import utils

freiburgforest_frq = []
weight_path = 'data/freiburgforest_5class_weight_med.txt'
with open(weight_path, 'r') as f:
    context = f.readlines()
    for x in context:
        x = x.strip().strip('\ufeff')
        freiburgforest_frq.append(float(x))
print("Number of class weights:", len(freiburgforest_frq))

parser = argparse.ArgumentParser(description='Multimodal Semantic Segmentation - ACNet training')
parser.add_argument('--train-dir', default=None, metavar='DIR',
                    help='path to train dataset')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=1500, type=int, metavar='N',
                    help='number of total epochs to run (default: 1500)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=5, type=int,
                    metavar='N', help='mini-batch size (default: 10)')
parser.add_argument('--lr', '--learning-rate', default=2e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--print-freq', '-p', default=500, type=int,
                    metavar='N', help='print batch frequency per steps (default: 500)')
parser.add_argument('--save-epoch-freq', '-s', default=100, type=int,
                    metavar='N', help='save epoch frequency (default: 100)')
parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--lr-decay-rate', default=0.8, type=float,
                    help='decay rate of learning rate (default: 0.8)')
parser.add_argument('--lr-epoch-per-decay', default=100, type=int,
                    help='epoch of per decay of learning rate (default: 150)')
parser.add_argument('--ckpt-dir', default='./model/', metavar='DIR',
                    help='path to save checkpoints')
parser.add_argument('--summary-dir', default='./summary', metavar='DIR',
                    help='path to save summary')
parser.add_argument('--checkpoint', action='store_true', default=False,
                    help='Using Pytorch checkpoint or not')

args = parser.parse_args()
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
image_w = 768
image_h = 384


def train():
    train_data = ACNet_data.FreiburgForest(
        transform=transforms.Compose([
            ACNet_data.scaleNorm(),
            ACNet_data.RandomScale((1.0, 1.4)),
            ACNet_data.RandomHSV((0.9, 1.1),
                                 (0.9, 1.1),
                                 (25, 25)),
            ACNet_data.RandomCrop(image_h, image_w),
            ACNet_data.RandomFlip(),
            ACNet_data.ToTensor(),
            ACNet_data.Normalize()
        ]),
        data_dir=args.train_dir
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=False)

    if args.last_ckpt:
        model = ACNet_models_V1.ACNet(num_class=5, pretrained=False)
    else:
        model = ACNet_models_V1.ACNet(num_class=5, pretrained=True)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.train()
    model.to(device)

    CEL_weighted = utils.CrossEntropyLoss2d(weight=freiburgforest_frq)
    CEL_weighted.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    global_step = 0

    if args.last_ckpt:
        global_step, args.start_epoch = utils.load_ckpt(model, optimizer, args.last_ckpt, device)

    lr_decay_lambda = lambda epoch: args.lr_decay_rate ** (epoch // args.lr_epoch_per_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_decay_lambda)

    writer = SummaryWriter(args.summary_dir)

    losses = []
    for epoch in tqdm(range(int(args.start_epoch), args.epochs)):

        if epoch % args.save_epoch_freq == 0 and epoch != args.start_epoch:
            utils.save_ckpt(args.ckpt_dir, model, optimizer, global_step, epoch)

        for batch_idx, sample in enumerate(train_loader):
            rgb = sample['rgb'].to(device)
            evi = sample['evi'].to(device)
            target_scales = [sample[s].to(device) for s in ['label', 'label2', 'label3', 'label4', 'label5']]

            optimizer.zero_grad()
            pred_scales = model(rgb, evi, args.checkpoint)
            loss = CEL_weighted(pred_scales, target_scales)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            global_step += 1
            if global_step % args.print_freq == 0 or global_step == 1:

                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.detach().cpu().numpy(), global_step, bins='doane')

                grid_image = make_grid(rgb[:3].detach().cpu(), 3, normalize=False)
                writer.add_image('RGB', grid_image, global_step)
                grid_image = make_grid(evi[:3].detach().cpu(), 3, normalize=False)
                writer.add_image('EVI', grid_image, global_step)
                grid_image = make_grid(utils.color_label(torch.argmax(pred_scales[0][:3], 1) + 1), 3, normalize=True,
                                       range=(0, 255))
                writer.add_image('Prediction', grid_image, global_step)
                grid_image = make_grid(utils.color_label(target_scales[0][:3]), 3, normalize=True, range=(0, 255))
                writer.add_image('GroundTruth', grid_image, global_step)
                writer.add_scalar('CrossEntropyLoss', loss.item(), global_step=global_step)
                writer.add_scalar('CrossEntropyLoss average', sum(losses) / args.print_freq, global_step=global_step)
                writer.add_scalar('Learning rate', scheduler.get_last_lr()[0], global_step=global_step)
                writer.add_scalar('Accuracy', utils.accuracy(
                    (torch.argmax(pred_scales[0], 1) + 1).detach().cpu().numpy().astype(int),
                    target_scales[0].detach().cpu().numpy().astype(int)), global_step=global_step)
                iou = utils.compute_IoU(
                    y_pred=(torch.argmax(pred_scales[0], 1) + 1).detach().cpu().numpy().astype(int),
                    y_true=target_scales[0].detach().cpu().numpy().astype(int),
                    num_classes=5
                )
                writer.add_scalar('IoU_Road', iou[0], global_step=global_step)
                writer.add_scalar('IoU_Grass', iou[1], global_step=global_step)
                writer.add_scalar('IoU_Vegetation', iou[2], global_step=global_step)
                writer.add_scalar('IoU_Sky', iou[3], global_step=global_step)
                writer.add_scalar('IoU_Obstacle', iou[4], global_step=global_step)
                writer.add_scalar('mIoU', np.mean(iou), global_step=global_step)

                losses = []

        scheduler.step()

    utils.save_ckpt(args.ckpt_dir, model, optimizer, global_step, args.epochs)
    print("Training completed ")


if __name__ == '__main__':
    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)
    if not os.path.exists(args.summary_dir):
        os.mkdir(args.summary_dir)
    train()

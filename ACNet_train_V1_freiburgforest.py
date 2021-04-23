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
# TODO: Try manually increasing the weights of uncommon classes (x2 for example)

parser = argparse.ArgumentParser(description='Multimodal Semantic Segmentation - ACNet training')
parser.add_argument('--train-dir', default=None, metavar='DIR',
                    help='path to train dataset')
parser.add_argument('--valid-dir', default=None, metavar='DIR',
                    help='path to valid dataset')
parser.add_argument('--train-dir2', default=None, metavar='DIR',
                    help='path to train dataset (2)')
parser.add_argument('--valid-dir2', default=None, metavar='DIR',
                    help='path to valid dataset (2)')
parser.add_argument('--modal1', default='rgb', help='Modality 1 for the model (3 channels)')
parser.add_argument('--modal2', default='evi2_gray', help='Modality 2 for the model (1 channel)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run (default: 1000)')
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
            ACNet_data.ScaleNorm(),
            ACNet_data.RandomRotate((-13, 13)),
            ACNet_data.RandomSkew((-0.05, 0.10)),
            ACNet_data.RandomScale((1.0, 1.4)),
            ACNet_data.RandomHSV((0.9, 1.1),
                                 (0.9, 1.1),
                                 (25, 25)),
            ACNet_data.RandomCrop(image_h, image_w),
            ACNet_data.RandomFlip(),
            ACNet_data.ToTensor(),
            ACNet_data.Normalize()
        ]),
        data_dirs=[args.train_dir, args.train_dir2],
        modal1_name=args.modal1,
        modal2_name=args.modal2,
    )

    valid_data = ACNet_data.FreiburgForest(
        transform=transforms.Compose([
            ACNet_data.ScaleNorm(),
            ACNet_data.ToTensor(),
            ACNet_data.Normalize()
        ]),
        data_dirs=[args.valid_dir],
        modal1_name=args.modal1,
        modal2_name=args.modal2,
    )

    '''
    # Split dataset into training and validation
    dataset_length = len(data)
    valid_split = 0.05  # tiny split due to the small size of the dataset
    valid_length = int(valid_split * dataset_length)
    train_length = dataset_length - valid_length
    train_data, valid_data = torch.utils.data.random_split(data, [train_length, valid_length])
    '''

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=False)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size * 3, shuffle=False,
                              num_workers=1, pin_memory=False)

    # Initialize model
    if args.last_ckpt:
        model = ACNet_models_V1.ACNet(num_class=5, pretrained=False)
    else:
        model = ACNet_models_V1.ACNet(num_class=5, pretrained=True)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.train()
    model.to(device)

    # Initialize criterion, optimizer and scheduler
    criterion = utils.CrossEntropyLoss2d(weight=freiburgforest_frq)
    criterion.to(device)

    # TODO: try with different optimizers and schedulers (CyclicLR exp_range for example)
    # TODO: try with a smaller LR (currently loss decay is too steep and then doesn't change)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    lr_decay_lambda = lambda epoch: args.lr_decay_rate ** (epoch // args.lr_epoch_per_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_decay_lambda)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=args.epochs // 2, T_mult=1, eta_min=6e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-4)
    global_step = 0

    # TODO: add early stop to avoid overfitting

    # Continue training from previous checkpoint
    if args.last_ckpt:
        global_step, args.start_epoch = utils.load_ckpt(model, optimizer, scheduler, args.last_ckpt, device)

    writer = SummaryWriter(args.summary_dir)
    losses = []
    for epoch in tqdm(range(int(args.start_epoch), args.epochs)):

        if epoch % args.save_epoch_freq == 0 and epoch != args.start_epoch:
            utils.save_ckpt(args.ckpt_dir, model, optimizer, scheduler, global_step, epoch)

        for batch_idx, sample in enumerate(train_loader):
            modal1, modal2 = sample['modal1'].to(device), sample['modal2'].to(device)
            target_scales = [sample[s].to(device) for s in ['label', 'label2', 'label3', 'label4', 'label5']]

            optimizer.zero_grad()
            pred_scales = model(modal1, modal2, args.checkpoint)
            loss = criterion(pred_scales, target_scales)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            global_step += 1
            if global_step % args.print_freq == 0 or global_step == 1:

                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.detach().cpu().numpy(), global_step, bins='doane')

                grid_image = make_grid(modal1[:3].detach().cpu(), 3, normalize=False)
                writer.add_image('Modal1', grid_image, global_step)
                grid_image = make_grid(modal2[:3].detach().cpu(), 3, normalize=False)
                writer.add_image('Modal2', grid_image, global_step)
                grid_image = make_grid(utils.color_label(torch.argmax(pred_scales[0][:3], 1) + 1), 3, normalize=True,
                                       range=(0, 255))
                writer.add_image('Prediction', grid_image, global_step)
                grid_image = make_grid(utils.color_label(target_scales[0][:3]), 3, normalize=True, range=(0, 255))
                writer.add_image('GroundTruth', grid_image, global_step)
                writer.add_scalar('Loss', loss.item(), global_step=global_step)
                writer.add_scalar('Loss average', sum(losses) / len(losses), global_step=global_step)
                writer.add_scalar('Learning rate', scheduler.get_last_lr()[0], global_step=global_step)

                # Compute validation metrics
                with torch.no_grad():
                    model.eval()

                    losses_val = []
                    acc_list = []
                    iou_list = []
                    for sample_val in valid_loader:
                        modal1_val, modal2_val = sample_val['modal1'].to(device), sample_val['modal2'].to(device)
                        target_val = sample_val['label'].to(device)
                        pred_val = model(modal1_val, modal2_val)

                        losses_val.append(criterion([pred_val], [target_val]).item())
                        acc_list.append(utils.accuracy(
                            (torch.argmax(pred_val, 1) + 1).detach().cpu().numpy().astype(int),
                            target_val.detach().cpu().numpy().astype(int))[0])
                        iou_list.append(utils.compute_IoU(
                            y_pred=(torch.argmax(pred_val, 1) + 1).detach().cpu().numpy().astype(int),
                            y_true=target_val.detach().cpu().numpy().astype(int),
                            num_classes=5
                        ))

                    writer.add_scalar('Loss validation', sum(losses_val) / len(losses_val), global_step=global_step)
                    writer.add_scalar('Accuracy', sum(acc_list) / len(acc_list), global_step=global_step)
                    iou = np.mean(np.stack(iou_list, axis=0), axis=0)
                    writer.add_scalar('IoU_Road', iou[0], global_step=global_step)
                    writer.add_scalar('IoU_Grass', iou[1], global_step=global_step)
                    writer.add_scalar('IoU_Vegetation', iou[2], global_step=global_step)
                    writer.add_scalar('IoU_Sky', iou[3], global_step=global_step)
                    writer.add_scalar('IoU_Obstacle', iou[4], global_step=global_step)
                    writer.add_scalar('mIoU', np.mean(iou), global_step=global_step)

                    model.train()

                losses = []

        scheduler.step()

    utils.save_ckpt(args.ckpt_dir, model, optimizer, scheduler, global_step, args.epochs)
    print("Training completed ")


if __name__ == '__main__':
    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)
    if not os.path.exists(args.summary_dir):
        os.mkdir(args.summary_dir)
    train()

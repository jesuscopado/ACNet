import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch import nn

med_frq = [2.422187, 0.888967, 0.573932, 1.000000, 70.225625]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

label_colours = [
    (255, 255, 255),  # void
    (170, 170, 170),  # Road
    (0, 255, 0),  # Grass
    (102, 102, 51),  # Vegetation and Tree
    (0, 120, 255),  # Sky
    (0, 0, 0)  # Obstacle
]


class CrossEntropyLoss2d_eval(nn.Module):
    def __init__(self, weight=med_frq):
        super(CrossEntropyLoss2d_eval, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(weight)).float(), reduction='none')

    def forward(self, inputs_scales, targets_scales):
        losses = []
        inputs = inputs_scales
        targets = targets_scales
        # for inputs, targets in zip(inputs_scales, targets_scales):
        mask = targets > 0
        targets_m = targets.clone()
        targets_m[mask] -= 1
        loss_all = self.ce_loss(inputs, targets_m.long())
        losses.append(torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float()))
        total_loss = sum(losses)
        return total_loss


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=med_frq):
        super(CrossEntropyLoss2d, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(weight)).float(), reduction='none')

    def forward(self, inputs_scales, targets_scales):
        losses = []
        for inputs, targets in zip(inputs_scales, targets_scales):
            mask = targets > 0
            targets_m = targets.clone()
            targets_m[mask] -= 1
            loss_all = self.ce_loss(inputs, targets_m.long())
            losses.append(torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float()))
        total_loss = sum(losses)
        return total_loss


# hxx add, focal loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.loss = nn.NLLLoss(weight=torch.from_numpy(np.array(weight)).float(),
                               size_average=self.size_average, reduce=False)

    def forward(self, input, target):
        return self.loss((1 - F.softmax(input, 1)) ** 2 * F.log_softmax(input, 1), target)


class FocalLoss2d(nn.Module):
    def __init__(self, weight=med_frq, gamma=0):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.fl_loss = FocalLoss(gamma=self.gamma, weight=self.weight, size_average=False)

    def forward(self, inputs_scales, targets_scales):
        losses = []
        for inputs, targets in zip(inputs_scales, targets_scales):
            mask = targets > 0
            targets_m = targets.clone()
            targets_m[mask] -= 1
            loss_all = self.fl_loss(inputs, targets_m.long())
            losses.append(torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float()))
        total_loss = sum(losses)
        return total_loss


def color_label_eval(label):
    # label = label.detach().cpu().numpy()
    label = label.astype(int)
    colored_label = np.vectorize(lambda x: label_colours[x])

    colored = np.asarray(colored_label(label)).astype(np.float32)
    colored = colored.squeeze()

    # return torch.from_numpy(colored.transpose([1, 0, 2, 3]))
    return colored


def color_label(label):
    label = label.detach().cpu().numpy().astype(int)
    colored_label = np.vectorize(lambda x: label_colours[x])

    colored = np.asarray(colored_label(label)).astype(np.float32)
    colored = colored.squeeze()

    try:
        return torch.from_numpy(colored.transpose([1, 0, 2, 3]))
    except ValueError:
        return torch.from_numpy(colored[np.newaxis, ...])


def print_log(global_step, epoch, local_count, count_inter, dataset_size, loss, time_inter):
    print('Step: {:>5} Train Epoch: {:>3} [{:>4}/{:>4} ({:3.1f}%)]    '
          'Loss: {:.6f} [{:.2f}s every {:>4} data]'.format(
        global_step, epoch, local_count, dataset_size,
        100. * local_count / dataset_size, loss.item(), time_inter, count_inter))


def save_ckpt_old(ckpt_dir, model, optimizer, global_step, epoch, local_count, num_train):
    # usually this happens only on the start of a epoch
    epoch_float = epoch + (local_count / num_train)
    state = {
        'global_step': global_step,
        'epoch': epoch_float,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    ckpt_model_filename = "ckpt_epoch_{:0.2f}.pth".format(epoch_float)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))


def save_ckpt(ckpt_dir, model, optimizer, scheduler, global_step, epoch):
    state = {
        'global_step': global_step,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    ckpt_model_filename = 'ckpt_model.pth'
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)


def load_ckpt(model, optimizer, scheduler, model_file, device):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        if device.type == 'cuda':
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_file, checkpoint['epoch']))
        step = checkpoint['global_step']
        epoch = checkpoint['epoch']
        return step, epoch
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        sys.exit()


# added by hxx for iou calculation
def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    # imPred += 1 # hxx
    # imLab += 1 # label 应该是不用加的
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    area_pred, _ = np.histogram(imPred, bins=numClass, range=(1, numClass))
    area_lab, _ = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return area_intersection, area_union


# https://stackoverflow.com/questions/31653576/how-to-calculate-the-mean-iu-score-in-image-segmentation
def compute_IoU(y_pred, y_true, num_classes):
    # ytrue, ypred is a flatten vector
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    current = confusion_matrix(y_true, y_pred, labels=range(1, num_classes + 1))
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / (union.astype(np.float32) + 1e-10)
    return IoU


def accuracy(preds, label):
    valid = (label > 0)  # hxx
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def macc(preds, label, num_class):
    a = np.zeros(num_class)
    b = np.zeros(num_class)
    for i in range(num_class):
        mask = (label == i + 1)
        a_sum = (mask * preds == i + 1).sum()
        b_sum = mask.sum()
        a[i] = a_sum
        b[i] = b_sum
    return a, b


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

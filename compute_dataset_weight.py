import argparse
import os

import imageio
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--labels-dir', default=None, metavar='DIR',
                    help='path to labels')
parser.add_argument('--output-file-inv', default='data/weight.txt',
                    help='path to file where to write the weights per class (inverse)')
parser.add_argument('--output-file-med', default='data/weight.txt',
                    help='path to file where to write the weights per class (median)')
args = parser.parse_args()


# https://github.com/ultralytics/yolov3/issues/249#L59
def labels_to_class_weights_inverse(labels, nc=5):
    # Get class weights (inverse frequency) from training labels
    weights = np.bincount(labels, minlength=nc).astype(float)  # occurrences per class
    weights /= weights.sum()  # normalize
    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    return weights


# https://arxiv.org/pdf/1511.00561.pdf
def labels_to_class_weights_median(labels, nc=5):
    class_freq = np.bincount(labels, minlength=nc).astype(float)  # occurrences per class
    median_freq = np.median(class_freq)  # median of class frequencies
    weights = median_freq / class_freq  # ratio of the median divided by the class frequency
    return weights


def main():
    labels = np.array([]).astype(np.uint8)
    for filename in os.listdir(args.labels_dir):
        label_path = os.path.join(args.labels_dir, filename)
        label = imageio.imread(label_path).flatten()
        labels = np.concatenate((labels, label))
    weight = labels_to_class_weights_inverse(labels - 1)
    np.savetxt(args.output_file_inv, weight, delimiter='\n', fmt='%.6f')
    med_weight = labels_to_class_weights_median(labels - 1)
    np.savetxt(args.output_file_med, med_weight, delimiter='\n', fmt='%.6f')


if __name__ == '__main__':
    main()

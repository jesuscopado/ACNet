import argparse
import os

import imageio
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--labels-dir', default=None, metavar='DIR',
                    help='path to labels')
parser.add_argument('--output-file', default='data/weight.txt',
                    help='path to file where to write the weights per class')
args = parser.parse_args()


# https://github.com/ultralytics/yolov3/issues/249#L59
def labels_to_class_weights(labels, nc=5):
    # Get class weights (inverse frequency) from training labels
    weights = np.bincount(labels, minlength=nc).astype(float)  # occurences per class
    weights /= weights.sum()  # normalize
    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    return weights


def main():
    labels = np.array([]).astype(np.uint8)
    for filename in os.listdir(args.labels_dir):
        label_path = os.path.join(args.labels_dir, filename)
        label = imageio.imread(label_path).flatten()
        labels = np.concatenate((labels, label))
    weight = labels_to_class_weights(labels)
    np.savetxt(args.output_file, weight, delimiter='\n', fmt='%.6f')


if __name__ == '__main__':
    main()

from __future__ import print_function

import json
import os
import h5py
import cPickle

DATAROOT = 'data/images/'
folder = {
    'train': 'train2014/',
    'valid': 'val2014/',
    'test': 'test2014/'
}
prefix = {
    'train': 'COCO_train2014_000000',
    'valid': 'COCO_val2014_000000',
    'test': 'COCO_test2014_000000',
}


def get_images(split):
    img_ids = []
    filenames = []
    directory = os.path.join(DATAROOT, folder[split])
    for parent, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if os.path.isfile(os.path.join(directory, filename)):
                print(filename, ':', int(filename[-6:]))
                img_ids.append(int(filename[-6:]))
                filenames.append(os.path.join(directory, filename))
    return img_ids, filenames


def get_all_images():
    train_ids, train_files = get_images('train')
    valid_ids, valid_files = get_images('valid')
    test_ids, test_files = get_images('test')


if __name__ == '__main__':
    get_all_images()

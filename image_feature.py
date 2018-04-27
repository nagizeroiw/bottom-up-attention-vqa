from __future__ import print_function

import json
import os
import h5py
import cPickle

'''
get_all_images():
    from downloaded COCO dataset to pickle files storing 
        split_train.pkl
        split_valid.pkl
        split_test.pkl
            {
                'image_ids': [0, ...]
                'file_names': ['data/images/COCO_train2014_000000000000.jpg', ...]
            }


'''

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
dataset_output_file = {
    'train': 'split_train.pkl',
    'valid': 'split_valid.pkl',
    'test': 'split_test.pkl'
}


def get_images(split):
    img_ids = []
    filenames = []
    id2file = {}
    directory = os.path.join(DATAROOT, folder[split])
    for parent, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if os.path.isfile(os.path.join(directory, filename)):
                img_id = int(filename.split('.')[0][-6:])
                # print(filename, ':', img_id)
                img_ids.append(img_id)
                filenames.append(os.path.join(directory, filename))
                id2file[img_id] = os.path.join(directory, filename)
    print('> find %d images for split %s.' % (len(img_ids), split))
    to_save = {
        'image_ids': img_ids,
        'file_names': filenames
    }
    cPickle.dump(to_save, open(os.path.join(DATAROOT, dataset_output_file[split]), 'w'))
    return id2file


def get_all_images():
    id2file = get_images('train')
    id2file.update(get_images('valid'))
    id2file.update(get_images('test'))
    print(len(id2file))
    cPickle.dump(id2file, open(os.path.join(DATAROOT, 'id2file.pkl'), 'w'))


if __name__ == '__main__':
    get_all_images()

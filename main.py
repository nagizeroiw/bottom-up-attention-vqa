import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset, VQAFeatureDatasetWithPair, VQAFeatureDatasetEnd2End
import base_model
from train import train, measure, seek
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='train', help='train|test|test-dev|seek-train|seek-valid|seek-test')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--cnn_model', type=str, default='resnet101')
    parser.add_argument('--model_root', type=str, default='data/cnn_weights/')
    parser.add_argument('--output', type=str, default='saved_models/with_log')
    parser.add_argument('--start_with', type=str, default=None)

    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
    parser.add_argument('--grad_clip_rate', type=float, default=0.5, help='grad clip threshold')
    parser.add_argument('--rnn_layer', type=int, default=1, help='number of rnn layers')

    parser.add_argument('--pair_loss_type', type=str, default='margin@repr', help='@att, @repr, margin@att, margin@repr')
    parser.add_argument('--pair_loss_weight', type=float, default=1e-4, help='alpha in pair loss')
    parser.add_argument('--gamma', type=float, default=2.5, help='margin threshold gamma for pair_loss_margin')

    parser.add_argument('--stackatt_nlayers', type=int, default=1, help='1|2|3')

    parser.add_argument('--train_dataset', type=str, default='pairwise', help='all|filter|pairwise|end2end|all_pair')
    parser.add_argument('--test_dataset', type=str, default='pairwise', help='all|filter|pairwise|end2end')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)

    if args.task == 'train':

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True

        start = time.time()
        batch_size = args.batch_size
        train_batch = batch_size
        test_batch = batch_size

        dictionary = Dictionary.load_from_file('data/dictionary.pkl')
        if args.train_dataset == 'all':
            train_dset = VQAFeatureDataset('train', dictionary, filter_pair=False)
        elif args.train_dataset == 'filter':
            train_dset = VQAFeatureDataset('train', dictionary, filter_pair=True)
        elif args.train_dataset == 'pairwise':
            train_batch = batch_size / 2
            train_dset = VQAFeatureDatasetWithPair('train', dictionary)
        elif args.train_dataset == 'end2end':
            train_dset = VQAFeatureDatasetEnd2End('train', dictionary, filter_pair=False)
        elif args.train_dataset == 'all_pair':
            train_dset_all = VQAFeatureDataset('train', dictionary, filter_pair=False)
            train_dset_pair = VQAFeatureDatasetWithPair('train', dictionary, preloaded=train_dset_all.pre_loaded())
            train_dset = train_dset_all  # for model building: dset.vdim and co.
        else:
            raise NotImplemented('dataset not implemented: %s' % args.train_dataset)

        if args.test_dataset == 'all':
            eval_dset = VQAFeatureDataset('val', dictionary, filter_pair=False)
        elif args.test_dataset == 'filter':
            eval_dset = VQAFeatureDataset('val', dictionary, filter_pair=True)
        elif args.test_dataset == 'pairwise':
            test_batch = batch_size / 2
            eval_dset = VQAFeatureDatasetWithPair('val', dictionary)
        elif args.test_dataset == 'end2end':
            eval_dset = VQAFeatureDatasetEnd2End('val', dictionary, filter_pair=False)
        else:
            raise NotImplemented('dataset not implemented: %s' % args.train_dataset)
            
        print '> data loaded. time: %.2fs' % (time.time() - start)

        constructor = 'build_%s' % args.model
        model = getattr(base_model, constructor)(train_dset, args.num_hid, args).cuda()
        model.w_emb.init_embedding('data/glove6b_init_300d.npy')
        model = nn.DataParallel(model).cuda()

        if args.train_dataset == 'all_pair':
            train_loader_all = DataLoader(train_dset_all, train_batch, shuffle=True, num_workers=1)
            train_loader_pair = DataLoader(train_dset_pair, train_batch / 2, shuffle=True, num_workers=1)
            train_loader = (train_loader_all, train_loader_pair)
        else:
            train_loader = DataLoader(train_dset, train_batch, shuffle=True, num_workers=1)
        eval_loader =  DataLoader(eval_dset, test_batch, shuffle=True, num_workers=1)

        train(model, train_loader, eval_loader, args)

    elif args.task.startswith('test'):

        torch.backends.cudnn.benchmark = True

        start = time.time()
        batch_size = args.batch_size

        dictionary = Dictionary.load_from_file('data/dictionary.pkl')
        test_dset = VQAFeatureDataset(args.task, dictionary, filter_pair=False, is_test=True)
            
        print '> data loaded. time: %.2fs' % (time.time() - start)

        constructor = 'build_%s' % args.model
        model = getattr(base_model, constructor)(test_dset, args.num_hid, args).cuda()
        model.w_emb.init_embedding('data/glove6b_init_300d.npy')
        model = nn.DataParallel(model).cuda()

        test_loader = DataLoader(test_dset, batch_size, shuffle=False, num_workers=1)

        measure(model, test_loader, args)

    elif args.task.startswith('seek'):

        split = args.task.split('-')[1]

        print('> seek on split %s' % split)

        torch.backends.cudnn.benchmark = True

        start = time.time()
        batch_size = 1

        dictionary = Dictionary.load_from_file('data/dictionary.pkl')
        test_dset = VQAFeatureDataset(split, dictionary, filter_pair=False, is_test=True)
            
        print '> data loaded. time: %.2fs' % (time.time() - start)

        constructor = 'build_%s' % args.model
        model = getattr(base_model, constructor)(test_dset, args.num_hid, args).cuda()
        model.w_emb.init_embedding('data/glove6b_init_300d.npy')
        model = model.cuda()

        test_loader = DataLoader(test_dset, batch_size, shuffle=False, num_workers=1)

        seek(model, test_loader, args)
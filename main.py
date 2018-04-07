import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset, VQAFeatureDatasetWithPair
import base_model
from train import train, measure
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='train', help='train|measure')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='saved_models/with_log')
    parser.add_argument('--start_with', type=str, default=None)

    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
    parser.add_argument('--grad_clip_rate', type=float, default=0.25, help='grad clip threshold')

    parser.add_argument('--pair_loss_type', type=str, default='margin@repr', help='@att, @repr, margin@att, margin@repr')
    parser.add_argument('--pair_loss_weight', type=float, default=1e-4, help='alpha in pair loss')
    parser.add_argument('--gamma', type=float, default=2.5, help='margin threshold gamma for pair_loss_margin')
    parser.add_argument('--use_pair', type=bool, default=True, help='whether use pair-wise batch feeding')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    start = time.time()

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    if args.use_pair:
        train_dset = VQAFeatureDatasetWithPair('train', dictionary)
    else:
        train_dset = VQAFeatureDataset('train', dictionary)
    eval_dset = VQAFeatureDataset('val', dictionary)
    batch_size = args.batch_size

    print '> data loaded. time: %.2fs' % (time.time() - start)

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid, args).cuda()
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')

    model = nn.DataParallel(model).cuda()

    train_loader = DataLoader(train_dset, batch_size / 2, shuffle=True, num_workers=1)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)
    if args.task == 'train':
        train(model, train_loader, eval_loader, args)
    elif args.task == 'measure':
        measure(model, train_loader, eval_loader, args)

import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
from progressbar import ProgressBar
import json
import cPickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

seen_loss_shape = False

try:
    import tensorflow as tf
except ImportError:
    print "! Tensorflow not installed; Not tensorboard logging."
    tf = None


def model_setting(args):
    setting = '< [%s][lr: %f][ploss: %s, %f, %f][seed: %d]>' % (
            args.output.split('/')[-1],
            args.lr,
            args.pair_loss_type,
            args.pair_loss_weight,
            args.gamma,
            args.seed
        )
    return setting


def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)


def instance_bce_with_logits(logits, labels, pair_loss=None, raw_pair_loss=None):
    assert logits.dim() == 2
    if labels.dim() == 3:  # handle data with pair
        batch_size, _, n_answers = labels.size()
        labels = labels.view(-1, n_answers)

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)

    if pair_loss is not None:

        # if random.randint(1, 100) == 1:
        #     print(loss.data[0], pair_loss.data[0], raw_pair_loss.data[0])

        loss += pair_loss

    return loss


def compute_score_with_logits(logits, labels):

    if labels.dim() == 3:  # handle data with pair
        batch_size, _, n_answers = labels.size()
        labels = labels.view(-1, n_answers)

    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels.data)
    return scores


def seek(model, test_set, args, split, question_id):

    image_path = {
        'train': 'train2014/COCO_train2014_000000',
        'val': 'val2014/COCO_val2014_000000',
        'test': 'test2015/COCO_test2015_000000'
    }

    image_root = './data/images/'

    # load from start_with
    assert args.start_with is not None
    print('> loading saved model from %s...' % os.path.join(args.start_with, 'model.pth'))
    model.load_state_dict(torch.load(os.path.join(args.start_with, 'model.pth')))
    model.train(False)
    print('> model loaded')

    label2ans_file = os.path.join('data/cache', 'trainval_label2ans.pkl')
    label2ans = cPickle.load(open(label2ans_file, 'rb'))

    v, b, q, qid = test_set.get_qid_minibatch(question_id)
    v = Variable(v).cuda()
    b = Variable(b).cuda()
    q = Variable(q).cuda()
    qid = Variable(qid).cuda()

    pred, att = model.module.seek(v, b, q, qid)
    logits = torch.max(pred, 1)[1].data  # argmax -> size (batch,)
    print(int(qid[0]), int(logits[0]), label2ans[int(logits[0])])

    iid = int(qid[0]) / 1000
    image_file_name = image_path[split] + '%06d.jpg' % iid
    print('image file name: %s' % image_file_name)
    # something like (375, 500, 3)

    with open(os.path.join(image_root, image_file_name)) as image_fp:
        image = plt.imread(image_fp)

    print('image shape', image.shape)
    h, w, _ = image.shape

    fig, ax = plt.subplots()
    ax.imshow(image)

    for k in xrange(36):

        weight = att[0, k].data[0]

        x1, y1, x2, y2, dx, dy = b[0, k].data

        x1, dx = x1 * w, dx * w
        y1, dy = y1 * h, dy * h

        rect = patches.Rectangle((x1, y1), dx, dy, edgecolor='black', facecolor='red', alpha=min(1, weight))
        
        if weight >= 0.1:
            plt.text(x1, y1, '%.2f' % weight, verticalalignment='top', horizontalalignment='left', color='black')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.xlabel('%s -> %s' % (args.start_with.split('/')[1], label2ans[int(logits[0])]))

    print('> saving image to %s...' % args.seek_output)
    plt.savefig(args.seek_output)


def measure(model, test_loader, args):

    # load from start_with
    assert args.start_with is not None
    print('> loading saved model from %s...' % os.path.join(args.start_with, 'model.pth'))
    model.load_state_dict(torch.load(os.path.join(args.start_with, 'model.pth')))
    model.train(False)
    print('> model loaded')

    label2ans_file = os.path.join('data/cache', 'trainval_label2ans.pkl')
    label2ans = cPickle.load(open(label2ans_file, 'rb'))

    all_results = {}

    print('> total questions: %d' % len(test_loader.dataset))

    disp_freq = len(test_loader) / 200

    for i, (v, b, q, qid) in enumerate(test_loader):
        v = Variable(v).cuda()
        b = Variable(b).cuda()
        q = Variable(q).cuda()
        qid = Variable(qid).cuda()

        pred, _, __ = model(v, b, q, qid)
        logits = torch.max(pred, 1)[1].data  # argmax -> size (batch,)

        if i % disp_freq == 0:
            print(int(qid[0]), int(logits[0]), label2ans[int(logits[0])])

        for k in range(logits.size()[0]):
            # print(int(qid[k]), int(logits[k]), label2ans[int(logits[k])])
            assert int(qid[k]) not in all_results
            all_results[int(qid[k])] = label2ans[int(logits[k])]

    results = []
    for qid, ans in all_results.iteritems():
        results.append({
            'question_id': qid,
            'answer': ans
            })
    with open('results.json', 'w') as fp:
        json.dump(results, fp)


def train(model, train_loader, eval_loader, args):

    num_epochs = args.epochs
    output = args.output
    lr = args.lr

    utils.create_dir(output)
    parameters = [param for param in model.parameters() if param.requires_grad is True]
    optim = torch.optim.Adamax(parameters, lr=lr)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    tf_writer = tf and tf.summary.FileWriter(os.path.join(output, 'tf_log/'))

    logger.write('> start training')
    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        total_pair_loss = 0
        total_raw_pair_loss = 0
        t = time.time()

        logger.write(model_setting(args))
        logger.write(str(args))

        # all&pair
        try:
            train_loader_all, train_loader_pair = train_loader
            if epoch < args.all_pair_d:
                dataloader = train_loader_pair
                print('> training with pairwise')
            else:
                dataloader = train_loader_all
                print('> training with all')
        except:
            dataloader = train_loader

        bar = ProgressBar(maxval=len(dataloader))
        bar.start()

        loss_len = 0.

        for i, items in enumerate(dataloader):
            '''
                v:features (b, 2, 36, 2048) -> image features (represented by 36 top objects / salient regions)
                b:spatials (b, 2, 36, 6) -> spatial features (() of 36 top objects)
                q:question (b, 2, 14) -> question sentence sequence (tokenized)
                a:target (b, 2, 3129) -> answer target (with soft labels)
            '''
            if args.train_dataset == 'end2end':
                v, q, a = items
                v = Variable(v).cuda()
                q = Variable(q).cuda()
                a = Variable(a).cuda()
                pred, pair_loss, raw_pair_loss = model(v, q, a)
            elif args.train_dataset == 'allpair':
                v, b, q, a = items
                v = Variable(v).cuda()
                b = Variable(b).cuda()
                q = Variable(q).cuda()
                a = Variable(a).cuda()
                pred, pair_loss, raw_pair_loss = model(v, b, q, a, force_without_ploss=True)
            else:
                v, b, q, a = items
                v = Variable(v).cuda()
                b = Variable(b).cuda()
                q = Variable(q).cuda()
                a = Variable(a).cuda()
                pred, pair_loss, raw_pair_loss = model(v, b, q, a)

            optim.zero_grad()
            loss = instance_bce_with_logits(pred, a, pair_loss, raw_pair_loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_rate)
            optim.step()

            batch_score = compute_score_with_logits(pred, a).sum()
            loss_len += pred.size(0)
            if pair_loss is None:
                total_loss += loss.item() * pred.size(0)
            else:  # v.dim() == 4
                total_loss += loss.item() * pred.size(0)
                total_pair_loss += pair_loss.item() * pred.size(0)
                total_raw_pair_loss += raw_pair_loss.item() * pred.size(0)
            try:
                train_score += batch_score.item()
            except:
                train_score += batch_score
            bar.update(i)

        bar.finish()

        total_loss /= loss_len
        total_pair_loss /= loss_len
        total_raw_pair_loss /= loss_len
        train_score = 100 * train_score / loss_len

        train_time = time.time()

        logger.write('> epoch %d, train time: %.2f' % (epoch, train_time - t))
        logger.write('\ttrain_loss: %.2f, train_pair_loss: %.7f, train_raw_pair_loss: %.7f, train_score: %.2f' % \
            (total_loss, total_pair_loss, total_raw_pair_loss, train_score))

        model.train(False)
        eval_score, eval_pair_loss, eval_raw_pair_loss = evaluate(model, eval_loader, args)
        model.train(True)

        logger.write('> validation time: %.2f' % (time.time() - train_time))
        logger.write('\tvalid score: %.2f, valid_pair_loss: %.7f, valid_raw_pair_loss: %.7f' % \
            (100 * eval_score, eval_pair_loss, eval_raw_pair_loss))


        add_summary_value(tf_writer, 'loss', total_loss, epoch)
        add_summary_value(tf_writer, 'train_score', train_score, epoch)
        add_summary_value(tf_writer, 'valid_score', 100 * eval_score, epoch)
        add_summary_value(tf_writer, 'pair_loss', total_pair_loss, epoch)
        add_summary_value(tf_writer, 'raw_pair_loss', total_raw_pair_loss, epoch)
        add_summary_value(tf_writer, 'valid_pair_loss', eval_pair_loss, epoch)
        add_summary_value(tf_writer, 'valid_raw_pair_loss', eval_raw_pair_loss, epoch)
        tf_writer.flush()

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score
        logger.write('\tbest score: %.2f' % (100 * best_eval_score))


def evaluate(model, dataloader, args):
    score = 0
    num_data = 0
    total_pair_loss = 0
    total_raw_pair_loss = 0
    loss_len = 0.
    for items in iter(dataloader):

        if args.test_dataset == 'end2end':
            v, q, a = items
            with torch.no_grad():
                v = Variable(v).cuda()
                q = Variable(q).cuda()
                a = Variable(a).cuda()
                pred, pair_loss, raw_pair_loss = model(v, q, a)
        else:
            v, b, q, a = items
            with torch.no_grad():
                v = Variable(v).cuda()
                b = Variable(b).cuda()
                q = Variable(q).cuda()
                a = Variable(a).cuda()
                pred, pair_loss, raw_pair_loss = model(v, b, q, a)

        batch_score = compute_score_with_logits(pred, a).sum()
        loss_len += v.size(0) * 2
        try:
            score += batch_score.item()
            # print('batch score: %.2f | pred.size(0):%.1f' % (batch_score.item(), pred.size(0)))
        except:
            score += batch_score
            # print('batch score: %.2f || pred.size(0): %.1f' % (batch_score, pred.size(0)))
        num_data += pred.size(0)
        if pair_loss is not None:
            total_pair_loss += pair_loss.item() * v.size(0) * 2
        if raw_pair_loss is not None:
            total_raw_pair_loss += raw_pair_loss.item() * v.size(0) * 2

    total_pair_loss /= loss_len
    total_raw_pair_loss /= loss_len
    score = score / num_data
    return score, total_pair_loss, total_raw_pair_loss

import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
from progressbar import ProgressBar
import random

seen_loss_shape = False

try:
    import tensorflow as tf
except ImportError:
    print "! Tensorflow not installed; Not tensorboard logging."
    tf = None


def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)


def instance_bce_with_logits(logits, labels, pair_loss=None, raw_pair_loss=None):
    assert logits.dim() == 2
    if labels.dim() == 3:  # handle data with pair
        _, batch_size, n_answers = labels.size()
        labels = labels.view(-1, n_answers)

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)

    if pair_loss is not None:

        if random.randint(1, 100) == 1:
            print(loss.data[0], pair_loss.data[0], raw_pair_loss.data[0])

        loss += pair_loss

    return loss


def compute_score_with_logits(logits, labels):

    if labels.dim() == 3:  # handle data with pair
        _, batch_size, n_answers = labels.size()
        labels = labels.view(-1, n_answers)

    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def train(model, train_loader, eval_loader, args):

    num_epochs = args.epochs
    output = args.output
    lr = args.lr

    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters(), lr=lr)
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

        bar = ProgressBar(maxval=len(train_loader))
        bar.start()

        for i, (v, b, q, a) in enumerate(train_loader):
            '''
                v:features (b, 2, 36, 2048) -> image features (represented by 36 top objects / salient regions)
                b:spatials (b, 2, 36, 6) -> spatial features (() of 36 top objects)
                q:question (b, 2, 14) -> question sentence sequence (tokenized)
                a:target (b, 2, 3129) -> answer target (with soft labels)
            '''
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()

            pred, pair_loss, raw_pair_loss = model(v, b, q, a)
            loss = instance_bce_with_logits(pred, a, pair_loss, raw_pair_loss)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            if v.dim() == 3:
                total_loss += loss.data[0] * v.size(0)
            else:  # v.dim() == 4
                total_loss += loss.data[0] * v.size(0) * 2
                total_pair_loss += pair_loss.data[0] * v.size(0) * 2
                total_raw_pair_loss += raw_pair_loss.data[0] * v.size(0) * 2
                # print(loss.data[0] * v.size(0) * 2, pair_loss.data[0] * v.size(0))
            train_score += batch_score
            bar.update(i)

        bar.finish()
        train_t = time.time() - t

        total_loss /= train_loader.dataset.loss_len()
        total_pair_loss /= train_loader.dataset.loss_len()
        total_raw_pair_loss /= train_loader.dataset.loss_len()
        train_score = 100 * train_score / train_loader.dataset.loss_len()
        model.train(False)
        eval_score = evaluate(model, eval_loader)
        model.train(True)

        total_time = time.time() - t

        logger.write('> epoch %d, time: %.2f (train %.2f eval %.2f)' % (epoch, total_time, train_t, total_time - train_t))
        logger.write('\ttrain_loss: %.2f, train_pair_loss: %.7f, train_raw_pair_loss: %.7f, train_score: %.2f' % (total_loss, total_pair_loss, total_raw_pair_loss, train_score))
        logger.write('\teval score: %.2f' % (100 * eval_score))

        add_summary_value(tf_writer, 'loss', total_loss, epoch)
        add_summary_value(tf_writer, 'train_score', train_score, epoch)
        add_summary_value(tf_writer, 'eval_score', 100 * eval_score, epoch)
        add_summary_value(tf_writer, 'pair_loss', total_pair_loss, epoch)
        tf_writer.flush()

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score
        logger.write('\tbest score: %.2f' % (100 * best_eval_score))


def evaluate(model, dataloader):
    score = 0
    num_data = 0
    for v, b, q, a in iter(dataloader):
        v = Variable(v, volatile=True).cuda()
        b = Variable(b, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        pred, _, __ = model(v, b, q, None)
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score
        num_data += pred.size(0)

    score = score / dataloader.dataset.loss_len()
    return score

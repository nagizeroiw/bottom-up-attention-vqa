import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
from progressbar import ProgressBar

try:
    import tensorflow as tf
except ImportError:
    print "! Tensorflow not installed; Not tensorboard logging."
    tf = None


def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary)


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def train(model, train_loader, eval_loader, num_epochs, output):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    tf_writer = tf and tf.summary.FileWriter(os.path.join(output, 'tf_log/'))

    logger.write('> start training')

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()

        bar = ProgressBar(maxval=len(train_loader))
        bar.start()

        for i, (v, b, q, a) in enumerate(train_loader):
            '''
                v:features (b, 36, 2048) -> image features (represented by 36 top objects / salient regions)
                b:spatials (b, 36, 6) -> spatial features (() of 36 top objects)
                q:question (b, 14) -> question sentence sequence (tokenized)
                a:target (b, 3129) -> answer target (with soft labels)
            '''
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()

            pred = model(v, b, q, a)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data[0] * v.size(0)
            train_score += batch_score
            bar.update(i)

        bar.finish()
        train_t = time.time() - t

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        model.train(False)
        eval_score, bound = evaluate(model, eval_loader)
        model.train(True)

        total_time = time.time() - t

        logger.write('> epoch %d, time: %.2f (train %.2f eval %.2f)' % (epoch, total_time, train_t, total_time - train_t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        add_summary_value(tf_writer, 'loss', total_loss, epoch)
        add_summary_value(tf_writer, 'train_score', train_score, epoch)
        add_summary_value(tf_writer, 'eval_score', 100 * eval_score, epoch)
        tf_writer.flush()

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score
        logger.write('\tbest score: %.2f' % (100 * best_eval_score))


def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    for v, b, q, a in iter(dataloader):
        v = Variable(v, volatile=True).cuda()
        b = Variable(b, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        pred = model(v, b, q, None)
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score
        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound

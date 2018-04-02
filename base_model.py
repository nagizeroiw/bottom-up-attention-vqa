from __future__ import print_function
import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
from torch.autograd import Variable


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, args):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.pair_loss_weight = args.pair_loss_weight

        self.seen_back2normal_shape = False

    def see(self, var, name):
        if not self.seen_back2normal_shape:
            print(name, var.size())

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, 2, num_objs, obj_dim]
        b: [batch, 2, num_objs, b_dim]
        q: [batch, 2, seq_length]
        labels: [batch, 2, n_ans]

        return: logits, not probs
        """

        if v.dim() == 4:  # handle pair loss
            batch, _, num_objs, obj_dim = v.size()
            _, __, ___, b_dim = b.size()
            _, __, seq_length = q.size()

            v = v.view(-1, num_objs, obj_dim)  # (2 * batch, num_objs, obj_dim)
            b = b.view(-1, num_objs, b_dim)  # (2 * batch, num_objs, b_dim)
            q = q.view(-1, seq_length)  # (2 * batch, seq_length)
            with_pair_loss = True
        else:
            with_pair_loss = False

        '''
        if not self.seen_back2normal_shape:
            print('v', v.size())
            print('b', b.size())
            print('q', q.size())
            self.seen_back2normal_shape = True
        '''

        w_emb = self.w_emb(q)  # preprocess question [2 * batch, seq_length, wemb_dim]
        q_emb = self.q_emb(w_emb)  # question representation [2 * batch, q_dim]

        att = self.v_att(v, q_emb)  # attention weight [2 * batch, num_objs, obj_dim]
        v_emb = (att * v).sum(1)  # attended feature vector [2 * batch, obj_dim]
        v_emb.retain_grad()

        q_repr = self.q_net(q_emb)  # question representation [2 * batch, num_hid]
        v_repr = self.v_net(v_emb)  # image representation [2 * batch, num_hid]
        joint_repr = q_repr * v_repr  # joint embedding (joint representation) [2 * batch, num_hid]

        logits = self.classifier(joint_repr)  # answer (answer probabilities) [2 * batch, n_answers]

        if with_pair_loss:
            '''
            ### pair_loss_1 (on joint representations)
            joint_repr = joint_repr.view(batch, 2, -1)  # [batch, 2, num_hid]
            joint_repr = joint_repr.transpose(1, 2)  # [batch, num_hid, 2]
            repr1, repr2 = joint_repr[:, :, 0], joint_repr[:, :, 1]  # [batch, num_hid] * 2

            joint_repr = joint_repr.transpose(1, 2).view(batch * 2, -1)  # [2 * batch, num_hid]
            pair_loss = -1. * self.pair_loss_weight * (repr1 - repr2).norm(dim=1)  # [batch,]
            '''

            '''
            ### pair_loss_2 (on attended image features)
            v_emb = v_emb.view(batch, 2, -1)  # [batch, 2, obj_dim]
            v_emb = v_emb.transpose(1, 2)  # [batch, obj_dim, 2]
            emb1, emb2 = v_emb[:, :, 0], v_emb[:, :, 1]  # [batch, obj_dim] * 2

            raw_pair_loss = (emb1 - emb2).norm(dim=1) # [batch,]
            pair_loss = -1. * self.pair_loss_weight * raw_pair_loss  # [batch,]            
            pair_loss = pair_loss.mean(dim=0, keepdim=True) # [1,]

            self.seen_back2normal_shape = True
            '''

            ### pair_loss_3 (max_margin)
            labels1, labels2 = labels[:, 0, :], labels[:, 1, :]  # [batch, n_ans] * 2

            logits = logits.view(batch, 2, -1)  # [batch, 2, n_ans]
            logits1, logits2 = logits[:, 0, :], logits[:, 1, :]  # [batch, n_ans] * 2

            logits1.backward(labels2, retain_graph=True)
            df2_1 = v_emb.grad  # [batch * 2, v_dim]
            df2_1 = df2_1.view(batch, 2, -1)[:, 0, :]  # [batch, v_dim]
            # print('df2_1', df2_1.size())
            self.zero_grad()

            logits1.backward(labels1, retain_graph=True)
            df1_1 = v_emb.grad  # [batch * 2, v_dim]
            df1_1 = df1_1.view(batch, 2, -1)[:, 0, :]  # [batch, v_dim]
            # print('df1_1', df1_1.size())
            self.zero_grad()

            logits2.backward(labels1, retain_graph=True)
            df1_2 = v_emb.grad  # [batch * 2, v_dim]
            df1_2 = df1_2.view(batch, 2, -1)[:, 1, :]  # [batch, v_dim]
            # print('df1_2', df2_1.size())
            self.zero_grad()

            logits2.backward(labels2, retain_graph=True)
            df2_2 = v_emb.grad  # [batch * 2, v_dim]
            df2_2 = df2_2.view(batch, 2, -1)[:, 1, :]  # [batch, v_dim]
            # print('df2_2', df2_1.size())
            self.zero_grad()

            # ~ 1e-3
            # print(torch.max(df2_1), torch.max(df1_1), torch.max(df1_2), torch.max(df2_2))

            logits = logits.view(batch * 2, -1)  # [batch * 2, n_ans]

            ################################################################################

            f_2_1 = logits1 * labels2 - logits1 * labels1  # [batch, n_ans]
            f_1_2 = logits2 * labels1 - logits2 * labels2  # [batch, n_ans]

            # ~ 1e-3
            # print(f_2_1.sum(dim=1).mean(dim=0).item())

            # ~ 5e-3
            # print((df2_1 - df1_1).norm(2, dim=1).mean(dim=0).item())

            pair_loss_1 = f_2_1.sum(dim=1) / ((df2_1 - df1_1).norm(2, dim=1) + 1e-8)  # [batch,]
            pair_loss_2 = f_1_2.sum(dim=1) / ((df1_2 - df2_2).norm(2, dim=1) + 1e-8)  # [batch,]

            pair_loss_1 = pair_loss_1.clamp(-50., 50.)
            pair_loss_2 = pair_loss_2.clamp(-50., 50.)
            # print(pair_loss_1.mean().item())

            self.seen_back2normal_shape = True
            raw_pair_loss = (pair_loss_1).mean()
            pair_loss = self.pair_loss_weight * raw_pair_loss

        if with_pair_loss:
            return logits, pair_loss, raw_pair_loss
        return logits, None, None


def build_baseline0(dataset, num_hid, args):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier, args)


def build_baseline0_newatt(dataset, num_hid, args):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.5)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.5)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier, args)

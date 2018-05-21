from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Attention, NewAttention, DualAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
from torch.autograd import Variable
import random
from backbone.resnet_utils import myResnet
import backbone.resnet as resnet


from torchvision import transforms as trn
preprocess = trn.Compose([
    # trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


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
        self.pair_loss_type = args.pair_loss_type
        self.gamma = args.gamma
        self.num_hid = args.num_hid

        self.seen_back2normal_shape = False

    def see(self, var, name):
        if not self.seen_back2normal_shape:
            print(name, var.size())

    def seek(self, v, b, q, labels):
        """Seek attention map

        v: [batch, num_objs(36), obj_dim(2048)]
        b: [batch, num_objs(36), b_dim(6)]
        q: [batch, seq_length(14)]
        labels: [batch, 2, n_ans(3129)]

        return: logits,          (batch, n_answers)
                att_weights,     (batch, num_objs)
        """

        w_emb = self.w_emb(q)  # preprocess question [batch, seq_length, wemb_dim]
        q_emb = self.q_emb(w_emb)  # question representation [batch, q_dim]

        att = self.v_att(v, q_emb)  # attention weight [batch, num_objs]
        v_emb = (att * v).sum(1)  # attended feature vector [batch, obj_dim]

        q_repr = self.q_net(q_emb)  # question representation [batch, num_hid]
        v_repr = self.v_net(v_emb)  # image representation [batch, num_hid]
        joint_repr = q_repr * v_repr  # joint embedding (joint representation) [batch, num_hid]

        logits = self.classifier(joint_repr)  # answer (answer probabilities) [batch, n_answers]

        return logits, att

    def forward(self, v, b, q, labels, force_without_ploss=False):
        """Forward

        v: [batch, 2, num_objs(36), obj_dim(2048)]
        b: [batch, 2, num_objs(36), b_dim(6)]
        q: [batch, 2, seq_length(14)]
        labels: [batch, 2, n_ans(3129)]

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

        if force_without_ploss:
            with_pair_loss = False


        w_emb = self.w_emb(q)  # preprocess question [2 * batch, seq_length, wemb_dim]
        q_emb = self.q_emb(w_emb)  # question representation [2 * batch, q_dim]

        att = self.v_att(v, q_emb)  # attention weight [2 * batch, num_objs]
        v_emb = (att * v).sum(1)  # attended feature vector [2 * batch, obj_dim]

        q_repr = self.q_net(q_emb)  # question representation [2 * batch, num_hid]
        v_repr = self.v_net(v_emb)  # image representation [2 * batch, num_hid]
        joint_repr = q_repr * v_repr  # joint embedding (joint representation) [2 * batch, num_hid]

        logits = self.classifier(joint_repr)  # answer (answer probabilities) [2 * batch, n_answers]

        self.see(logits, 'logits')

        if with_pair_loss:

            ### no pair_loss (but use pair-wise training)
            if self.pair_loss_type == 'none':
                with_pair_loss = False

            ### pair_loss_2 (@attended image feature)
            elif self.pair_loss_type == '@att':
                v_emb = v_emb.view(batch, 2, -1)  # [batch, 2, obj_dim]
                v_emb = v_emb.transpose(1, 2)  # [batch, obj_dim, 2]
                emb1, emb2 = v_emb[:, :, 0], v_emb[:, :, 1]  # [batch, obj_dim] * 2

                raw_pair_loss = (emb1 - emb2).norm(2, dim=1) # [batch,]
                pair_loss = -1. * self.pair_loss_weight * raw_pair_loss  # [batch,]            
                pair_loss = pair_loss.mean(dim=0, keepdim=False) # [,]
                raw_pair_loss = raw_pair_loss.mean(dim=0, keepdim=False)

                self.seen_back2normal_shape = True

            ### pair_loss_1 (@joint representation)
            elif self.pair_loss_type == '@repr':
                joint_repr = joint_repr.view(batch, 2, -1)  # [batch, 2, num_hid]
                joint_repr = joint_repr.transpose(1, 2)  # [batch, num_hid, 2]
                repr1, repr2 = joint_repr[:, :, 0], joint_repr[:, :, 1]  # [batch, num_hid] * 2

                joint_repr = joint_repr.transpose(1, 2).view(batch * 2, -1)  # [2 * batch, num_hid]
                raw_pair_loss = (repr1 - repr2).norm(2, dim=1)
                pair_loss = -1. * self.pair_loss_weight * raw_pair_loss  # [,]
                pair_loss = pair_loss.mean(dim=0, keepdim=False)
                raw_pair_loss = raw_pair_loss.mean(dim=0, keepdim=False)

                self.seen_back2normal_shape = True

            ### pair_loss_3 (max-margin style pair loss)
            elif self.pair_loss_type.startswith('margin'):
                labels1, labels2 = labels[:, 0, :], labels[:, 1, :]  # [batch, n_ans] * 2

                logits = logits.view(batch, 2, -1)  # [batch, 2, n_ans]
                logits1, logits2 = logits[:, 0, :], logits[:, 1, :]  # [batch, n_ans] * 2

                if self.pair_loss_type == 'margin@att':
                    df_size = obj_dim
                elif self.pair_loss_type == 'margin@repr':
                    df_size = self.num_hid
                elif self.pair_loss_type == 'margin@jrepr':
                    df_size = self.num_hid

                df2_1 = torch.FloatTensor(batch, df_size)
                df1_1 = torch.FloatTensor(batch, df_size)
                df1_2 = torch.FloatTensor(batch, df_size)
                df2_2 = torch.FloatTensor(batch, df_size)

                df2_1 = Variable(df2_1, requires_grad=False).cuda()
                df1_1 = Variable(df1_1, requires_grad=False).cuda()
                df1_2 = Variable(df1_2, requires_grad=False).cuda()
                df2_2 = Variable(df2_2, requires_grad=False).cuda()

                if self.pair_loss_type == 'margin@att':
                    v_emb.retain_grad()

                    logits1.backward(labels2, retain_graph=True)
                    df2_1[:, :] = v_emb.grad.view(batch, 2, -1)[:, 0, :]  # [batch, v_dim]
                    v_emb.grad.zero_()
                    self.zero_grad()

                    logits1.backward(labels1, retain_graph=True)
                    df1_1[:, :] = v_emb.grad.view(batch, 2, -1)[:, 0, :]  # [batch, v_dim]
                    v_emb.grad.zero_()
                    self.zero_grad()

                    logits2.backward(labels1, retain_graph=True)
                    df1_2[:, :] = v_emb.grad.view(batch, 2, -1)[:, 1, :]  # [batch, v_dim]
                    v_emb.grad.zero_()
                    self.zero_grad()

                    logits2.backward(labels2, retain_graph=True)
                    df2_2[:, :] = v_emb.grad.view(batch, 2, -1)[:, 1, :]  # [batch, v_dim]
                    v_emb.grad.zero_()
                    self.zero_grad()
                elif self.pair_loss_type == 'margin@repr':
                    v_repr.retain_grad()

                    logits1.backward(labels2, retain_graph=True)
                    df2_1[:, :] = v_repr.grad.view(batch, 2, -1)[:, 0, :]  # [batch, v_dim]
                    v_repr.grad.zero_()
                    self.zero_grad()

                    logits1.backward(labels1, retain_graph=True)
                    df1_1[:, :] = v_repr.grad.view(batch, 2, -1)[:, 0, :]  # [batch, v_dim]
                    v_repr.grad.zero_()
                    self.zero_grad()

                    logits2.backward(labels1, retain_graph=True)
                    df1_2[:, :] = v_repr.grad.view(batch, 2, -1)[:, 1, :]  # [batch, v_dim]
                    v_repr.grad.zero_()
                    self.zero_grad()

                    logits2.backward(labels2, retain_graph=True)
                    df2_2[:, :] = v_repr.grad.view(batch, 2, -1)[:, 1, :]  # [batch, v_dim]
                    v_repr.grad.zero_()
                    self.zero_grad()
                elif self.pair_loss_type == 'margin@jrepr':
                    joint_repr.retain_grad()

                    logits1.backward(labels2, retain_graph=True)
                    df2_1[:, :] = joint_repr.grad.view(batch, 2, -1)[:, 0, :]  # [batch, v_dim]
                    joint_repr.grad.zero_()
                    self.zero_grad()

                    logits1.backward(labels1, retain_graph=True)
                    df1_1[:, :] = joint_repr.grad.view(batch, 2, -1)[:, 0, :]  # [batch, v_dim]
                    joint_repr.grad.zero_()
                    self.zero_grad()

                    logits2.backward(labels1, retain_graph=True)
                    df1_2[:, :] = joint_repr.grad.view(batch, 2, -1)[:, 1, :]  # [batch, v_dim]
                    joint_repr.grad.zero_()
                    self.zero_grad()

                    logits2.backward(labels2, retain_graph=True)
                    df2_2[:, :] = joint_repr.grad.view(batch, 2, -1)[:, 1, :]  # [batch, v_dim]
                    joint_repr.grad.zero_()
                    self.zero_grad()

                logits = logits.view(batch * 2, -1)  # [batch * 2, n_ans]

                ##### computing pair loss
                f_2_1 = logits1 * labels2 - logits1 * labels1  # [batch, n_ans]
                f_1_2 = logits2 * labels1 - logits2 * labels2  # [batch, n_ans]

                mk1 = df2_1 - df1_1
                mk2 = df1_2 - df2_2

                pair_loss_1 = f_2_1.sum(dim=1) / (mk1.norm(2, dim=1) + 1e-8)  # [batch,]
                pair_loss_2 = f_1_2.sum(dim=1) / (mk2.norm(2, dim=1) + 1e-8)  # [batch,]

                pair_loss_1 = (pair_loss_1 + self.gamma).clamp(min=0.)
                pair_loss_2 = (pair_loss_2 + self.gamma).clamp(min=0.)

                
                raw_pair_loss = (pair_loss_1 + pair_loss_2).mean(dim=0)
                pair_loss = self.pair_loss_weight * raw_pair_loss

        self.seen_back2normal_shape = True
        if with_pair_loss:
            return logits, pair_loss, raw_pair_loss
        return logits, None, None


class BaseModelStackAtt(nn.Module):
    def __init__(self, w_emb, q_emb, v_atts, q_net, v_net, query_net, classifier, args):
        super(BaseModelStackAtt, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_atts
        self.q_net = q_net
        self.v_net = v_net
        self.query_net = query_net
        self.classifier = classifier
        self.pair_loss_weight = args.pair_loss_weight
        self.pair_loss_type = args.pair_loss_type
        self.gamma = args.gamma
        self.num_hid = args.num_hid

        self.stackatt_nlayers = args.stackatt_nlayers

        self.seen_back2normal_shape = False

    def see(self, var, name):
        if not self.seen_back2normal_shape:
            print(name, var.size())

    def forward(self, v, b, q, labels, force_without_ploss=False):
        """Forward

        v: [batch, 2, num_objs(36), obj_dim(2048)]
        b: [batch, 2, num_objs(36), b_dim(6)]
        q: [batch, 2, seq_length(14)]
        labels: [batch, 2, n_ans(3129)]

        return: logits, not probs
        """


        if v.dim() == 4:  # handle pair loss
            batch, _, num_objs, obj_dim = v.size()
            _, __, ___, b_dim = b.size()
            _, __, seq_length = q.size()

            v = v.view(-1, num_objs, obj_dim)  # (batch, num_objs, obj_dim)
            b = b.view(-1, num_objs, b_dim)  # (batch, num_objs, b_dim)
            q = q.view(-1, seq_length)  # (batch, seq_length)
            with_pair_loss = True
        else:
            with_pair_loss = False

        if force_without_ploss:
            with_pair_loss = False

        w_emb = self.w_emb(q)  # preprocess question [batch, seq_length, wemb_dim]
        q_emb = self.q_emb(w_emb)  # question representation [batch, q_dim]

        query1 = torch.cat((v.mean(1), q_emb), 1)  # 1st attention query [batch, q_dim+obj_dim]
        keep1 = self.v_att.keep_prob(v, query1) # keep prob [batch, num_objs]
        att1 = nn.functional.softmax(keep1, dim=1)
        v_emb1 = (att1 * v).sum(1)  # 1st-attended feature vector [batch, obj_dim]
        v_emb = v_emb1

        if self.stackatt_nlayers > 1:

            query2 = torch.cat((v_emb1, q_emb), 1) # 2nd-attention query [batch, q_dim + obj_dim]
            keep2 = self.v_att.keep_prob(v, query2)
            att2 = nn.functional.softmax(keep1 * keep2, dim=1)
            v_emb2 = (att2 * v).sum(1)  # 2nd-attended feature vector [batch, obj_dim]

            v_emb = v_emb2

        if self.stackatt_nlayers > 2:

            query3 = torch.cat((v_emb2, q_emb), 1) # 3st-attention query [batch, q_dim + obj_dim]
            keep3 = self.v_att.keep_prob(v, query3)
            att3 = nn.functional.softmax(keep1 * keep2 * keep3, dim=1)
            v_emb3 = (att3 * v).sum(1)

            v_emb = v_emb3

        q_repr = self.q_net(q_emb)  # question representation [batch, num_hid]
        v_repr = self.v_net(v_emb)  # image representation [batch, num_hid]
        joint_repr = q_repr * v_repr  # joint embedding (joint representation) [batch, num_hid]

        logits = self.classifier(joint_repr)  # answer (answer probabilities) [batch, n_answers]

        self.seen_back2normal_shape = True
        return logits, None, None


class BaseModelWithCNN(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, cnn, args):
        super(BaseModelWithCNN, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.cnn = cnn
        self.pair_loss_weight = args.pair_loss_weight
        self.pair_loss_type = args.pair_loss_type
        self.gamma = args.gamma
        self.num_hid = args.num_hid

        self.seen_back2normal_shape = False

    def see(self, var, name):
        if not self.seen_back2normal_shape:
            print(name, var.size())

    def forward(self, v, q, labels):
        """Forward

        v: [batch, 3, 299, 299]
        q: [batch, seq_length(14)]
        labels: [batch, n_ans(3129)]

        return: logits, not probs
        """
        _, v = self.cnn(v)

        w_emb = self.w_emb(q)  # preprocess question [2 * batch, seq_length, wemb_dim]
        q_emb = self.q_emb(w_emb)  # question representation [2 * batch, q_dim]

        att = self.v_att(v, q_emb)  # attention weight [2 * batch, num_objs, obj_dim]
        v_emb = (att * v).sum(1)  # attended feature vector [2 * batch, obj_dim]

        q_repr = self.q_net(q_emb)  # question representation [2 * batch, num_hid]
        v_repr = self.v_net(v_emb)  # image representation [2 * batch, num_hid]
        joint_repr = q_repr * v_repr  # joint embedding (joint representation) [2 * batch, num_hid]

        logits = self.classifier(joint_repr)  # answer (answer probabilities) [2 * batch, n_answers]

        self.seen_back2normal_shape = True

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


def build_dualatt(dataset, num_hid, args):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.4)
    q_emb = QuestionEmbedding(300, num_hid, args.rnn_layer, False, 0.4)
    v_att = DualAttention(dataset.v_dim, q_emb.num_hid, num_hid, 0.2)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)

    model = BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier, args)
    return model


def build_stackatt(dataset, num_hid, args):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.4)
    q_emb = QuestionEmbedding(300, num_hid, args.rnn_layer, False, 0.4)
    v_att = NewAttention(dataset.v_dim, num_hid, num_hid, 0.2)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    query_net = FCNet([dataset.v_dim, num_hid])

    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)

    model = BaseModelStackAtt(w_emb, q_emb, v_att, q_net, v_net, query_net, classifier, args)
    return model


def build_stackdualatt(dataset, num_hid, args):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.4)
    q_emb = QuestionEmbedding(300, num_hid, args.rnn_layer, False, 0.4)
    v_att1 = DualAttention(dataset.v_dim, q_emb.num_hid, num_hid, 0.2)
    v_att2 = DualAttention(dataset.v_dim, q_emb.num_hid + num_hid, num_hid, 0.2)
    v_att3 = DualAttention(dataset.v_dim, q_emb.num_hid + num_hid, num_hid, 0.2)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    query_net = FCNet([dataset.v_dim, num_hid])

    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)

    model = BaseModelStackAtt(w_emb, q_emb, (v_att1, v_att2, v_att3), q_net, v_net, query_net, classifier, args)
    return model


def build_fine(dataset, num_hid, args):

    cnn = getattr(resnet, args.cnn_model)()
    cnn.load_state_dict(torch.load(os.path.join(args.model_root, args.cnn_model + '.pth')))
    my_cnn = myResnet(cnn)

    for param in my_cnn.parameters():
        param.requires_grad = False
    for param in my_cnn.resnet.layer4.parameters():
        param.requires_grad = True

    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.4)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.4)
    v_att = DualAttention(dataset.v_dim, q_emb.num_hid, num_hid, 0.2)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)

    model = BaseModelWithCNN(w_emb, q_emb, v_att, q_net, v_net, classifier, my_cnn, args)
    return model

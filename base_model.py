import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.seen_back2normal_shape = False

    def forward(self, v, b, q, labels):
        """Forward
    (past)
        # v: [batch, num_objs, obj_dim]
        # b: [batch, num_objs, b_dim]
        # q: [batch_size, seq_length]
    (now)
        v: [batch, 2, num_objs, obj_dim]
        b: [batch, 2, num_objs, b_dim]
        q: [batch, 2, seq_length]

        return: logits, not probs
        """

        batch, _, num_objs, obj_dim = v.size()
        _, __, ___, b_dim = b.size()
        _, __, seq_length = q.size()

        v = v.view(-1, num_objs, obj_dim)  # (2 * batch, num_objs, obj_dim)
        b = b.view(-1, num_objs, b_dim)  # (2 * batch, num_objs, b_dim)
        q = q.view(-1, seq_length)  # (2 * batch, seq_length)

        '''
        if not self.seen_back2normal_shape:
            print('v', v.size())
            print('b', b.size())
            print('q', q.size())
            self.seen_back2normal_shape = True
        '''

        w_emb = self.w_emb(q)  # preprocess question [batch, seq_length, wemb_dim]
        q_emb = self.q_emb(w_emb)  # question representation [batch, q_dim]

        att = self.v_att(v, q_emb)  # attention weight [batch, num_objs, obj_dim]
        v_emb = (att * v).sum(1)  # attended feature vector [batch, obj_dim]

        q_repr = self.q_net(q_emb)  # question representation [batch, num_hid]
        v_repr = self.v_net(v_emb)  # image representation [batch, num_hid]
        joint_repr = q_repr * v_repr  # joint embedding (joint representation) [batch, num_hid]
        logits = self.classifier(joint_repr)  # answer (answer probabilities) [batch, n_answers]
        return logits


def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)


def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.5)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.5)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)

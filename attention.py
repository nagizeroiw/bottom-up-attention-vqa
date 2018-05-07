import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from fc import FCNet


class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(Attention, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid])
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, dim=1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits


class NewAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(NewAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.on_repr= FCNet([num_hid, num_hid])
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, dim=1)
        return w

    def keep_prob(self, v, q):
        logits = self.logits(v, q)
        p = nn.functional.sigmoid(logits)
        return p

    def logits(self, v, q):
        batch, k, _ = v.size()

        #################### tanh
        v_proj = self.v_proj(v) # [batch, k, qdim]
        v_proj = self.dropout(v_proj)
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        q_proj = self.q_proj(q_proj)

        joint_repr = v_proj * q_proj  # was cat[v, q]
        joint_repr = self.dropout(joint_repr)
        joint_repr = self.on_repr(joint_repr)
        
        logits = self.linear(joint_repr)
        return logits


class DualAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(DualAttention, self).__init__()

        self.v_proj1 = FCNet([v_dim, num_hid])
        self.v_proj2 = FCNet([v_dim, num_hid])
        self.q_proj1 = FCNet([q_dim, num_hid])
        self.q_proj2 = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.on_repr1 = FCNet([num_hid, num_hid])
        self.on_repr2 = FCNet([num_hid, num_hid])
        self.linear1 = weight_norm(nn.Linear(num_hid, 1), dim=None)
        self.linear2 = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits1, logits2 = self.logits(v, q)
        w1 = nn.functional.softmax(logits1, dim=1)
        w2 = nn.functional.softmax(logits2, dim=1)
        return w1 + w2

    def logits(self, v, q):
        batch, k, _ = v.size()

        #################### tanh
        v_proj1 = self.v_proj1(v) # [batch, k, qdim]
        q_proj1 = self.q_proj1(q).unsqueeze(1).repeat(1, k, 1)
        v_proj1 = self.dropout(v_proj1)
        q_proj1 = self.dropout(q_proj1)

        v_proj2 = self.v_proj2(v)
        q_proj2 = self.q_proj2(q).unsqueeze(1).repeat(1, k, 1)
        v_proj2 = self.dropout(v_proj2)
        q_proj2 = self.dropout(q_proj2)

        joint_repr1 = v_proj1 * q_proj1  # was cat[v, q]
        joint_repr1 = self.dropout(joint_repr1)
        joint_repr1 = self.on_repr1(joint_repr1)

        joint_repr2 = v_proj2 * q_proj2  # was cat[v, q]
        joint_repr2 = self.dropout(joint_repr2)
        joint_repr2 = self.on_repr2(joint_repr2)

        logits1 = self.linear1(joint_repr1)
        logits2 = self.linear2(joint_repr2)
        return logits1, logits2

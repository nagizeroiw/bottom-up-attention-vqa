from __future__ import print_function
import os
import json
import cPickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,  # idx for hdf5 file
        'question'    : question['question'],
        'answer'      : answer}
    return entry


def _load_dataset(dataroot, name, img_id2val):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(
        dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    entries = []

    qid2eid = {}

    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        entries.append(_create_entry(img_id2val[img_id], question, answer))
        qid2eid[question['question_id']] = len(entries) - 1

    return entries, qid2eid


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, dataroot='data'):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val']

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        print('> num_ans_candidates:', self.num_ans_candidates)

        self.dictionary = dictionary

        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name)))

        print('> loading complementary pairs file')
        self.pairs = json.load(open(os.path.join(dataroot, 'v2_mscoco_%s2014_complementary_pairs.json' % name), 'r'))
        # train 200394 pairs, valid 95144 pairs
        # train 443757 questions, valid 214354 questions

        print('> loading features from h5 file')
        h5_path = os.path.join(dataroot, '%s36.hdf5' % name)
        with h5py.File(h5_path, 'r') as hf:
            # self.features = np.array(hf.get('image_features'))
            # self.spatials = np.array(hf.get('spatial_features'))
            self.features = hf.get('image_features')[:]
            self.spatials = hf.get('spatial_features')[:]
        print('> features.shape', self.features.shape)
        # train (82783, 36, 2048), val (40504, 36, 2048)
        print('> spatials.shape', self.spatials.shape)
        # train (82783, 36, 6), val (40504, 36, 6)

        self.entries, self.qid2eid = _load_dataset(dataroot, name, self.img_id2idx)

        self.tokenize()
        self.tensorize()
        self.v_dim = self.features.size(2)  # 2048
        self.s_dim = self.spatials.size(2)  # 6
        print('> features and labels loaded.')

        self.seen_pshape = False

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        self.features = torch.from_numpy(self.features)
        self.spatials = torch.from_numpy(self.spatials)

        seen_shape = False

        for entry in self.entries:
            question = np.array(entry['q_token'])
            if not seen_shape:
                print('> question.shape', question.shape)
                seen_shape = True
            question = torch.from_numpy(question)
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        features = self.features[entry['image']]
        spatials = self.spatials[entry['image']]

        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        return features, spatials, question, target
        # features (36, 2048) -> image features (represented by 36 top objects / salient regions)
        # spatials (36, 6) -> spatial features (() of 36 top objects)
        # question (14,) -> question sentence sequence (tokenized)
        # target (3129,) -> answer target (with soft labels)

    def __len__(self):
        return len(self.entries)

class VQAFeatureDatasetWithPair(VQAFeatureDataset):

    def __init__(self, name, dictionary, dataroot='data'):
            super(VQAFeatureDataset, self).__init__()

    def __getitem__(self, index):
        qid1, qid2 = self.pairs[index]
        ent1, ent2 = self.entries[self.qid2eid[qid1]], self.entries[self.qid2eid[qid2]]
        features1, features2 = self.features[ent1['image']], self.features[ent2['image']]
        spatials1, spatials2 = self.spatials[ent1['image']], self.spatials[ent2['image']]

        question1, question2 = ent1['q_token'], ent2['q_token']
        answer1, answer2 = ent1['answer'], ent2['answer']
        labels1, labels2 = answer1['labels'], answer2['labels']
        scores1, scores2 = answer1['scores'], answer2['scores']
        target1, target2 = torch.zeros(self.num_ans_candidates), torch.zeros(self.num_ans_candidates)
        if labels1 is not None:
            target1.scatter_(0, labels1, scores1)
        if labels2 is not None:
            target2.scatter_(0, labels2, scores2)

        p_features = torch.stack([features1, features2], dim=0)
        p_spatials = torch.stack([spatials1, spatials2], dim=0)
        p_question = torch.stack([question1, question2], dim=0)
        p_target = torch.stack([target1, target2], dim=0)

        # p_features (2, 36, 2048)
        # p_spatials (2, 36, 6)
        # p_question (2, 14)
        # p_target (2, 3129)
        return p_features, p_spatials, p_question, p_target
       

    def __len__(self):
        # return len(self.entries)
        return len(self.pairs)
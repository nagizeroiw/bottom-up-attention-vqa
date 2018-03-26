import json
import sys
import os
import random

data_root = './data/'

questions_file = {
    'train': 'v2_OpenEnded_mscoco_train2014_questions.json',
    'valid': 'v2_OpenEnded_mscoco_val2014_questions.json'
}

annotations_file = {
    'train': 'v2_mscoco_train2014_annotations.json',
    'valid': 'v2_mscoco_val2014_annotations.json'
}

complementary_pairs_file = {
    'train': 'v2_mscoco_train2014_complementary_pairs.json',
    'valid': 'v2_mscoco_val2014_complementary_pairs.json'
}

def process(split):
    questions = json.load(open(os.path.join(data_root, questions_file[split])))['questions']
    annotations = json.load(open(os.path.join(data_root, annotations_file[split])))['annotations']
    pairs = json.load(open(os.path.join(data_root, complementary_pairs_file[split])))

    q_id_1, q_id_2 = random.choice(pairs)

    print('> from spilt %s:' % split)
    print('> question id pair: (%d, %d)' % (q_id_1, q_id_2))

    i_id_1, i_id_2 = None, None
    q_str_1, q_str_2 = None, None
    ans_1, ans_2 = None, None

    for q in questions:
        if q['question_id'] == q_id_1:
            i_id_1 = q['image_id']
            q_str_1 = q['question']
        if q['question_id'] == q_id_2:
            i_id_2 = q['image_id']
            q_str_2 = q['question']

    for a in annotations:
        if a['question_id'] == q_id_1:
            assert a['image_id'] == i_id_1
            ans_1 = a['multiple_choice_answer']
        if a['question_id'] == q_id_2:
            assert a['image_id'] == i_id_2
            ans_2 = a['multiple_choice_answer']

    print('> image id pair: (%d, %d)' % (i_id_1, i_id_2))
    print('> question 1: %s' % q_str_1)
    print('> answer 1: %s' % ans_1)
    print('> question 2: %s' % q_str_2)
    print('> answer 2: %s' % ans_2)


if __name__ == '__main__':
    try:
        split = sys.argv[1]
    except KeyError:
        split = 'train'
    process(split)

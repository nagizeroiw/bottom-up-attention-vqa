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

image_path = {
    'train': 'train2014/COCO_train2014_000000',
    'valid': 'val2014/COCO_val2014_000000'
}


def check(split):
    questions = json.load(open(os.path.join(data_root, questions_file[split])))['questions']
    annotations = json.load(open(os.path.join(data_root, annotations_file[split])))['annotations']

    qid2ques = {}
    qid2ans = {}
    for q in questions:
        qid = q['question_id']
        ques = q['question']
        qid2ques[qid] = ques
    for a in annotations:
        qid = a['question_id']
        image_id = a['image_id']
        ans = a['multiple_choice_answer']
        qid2ans[qid] = (image_id, ans)

    while True:
        qid = raw_input('> ')
        qid = int(qid)
        if qid not in qid2ans:
            print('! cannot find question id: %s' % qid)
            continue
        image_id, ans = qid2ans[qid]

        file_name = image_path[split] + '%06d.jpg' % int(image_id)
        print('file name:', file_name)
        os.system('scp %s ./vis/' % ('jungpu6:~/vqa-butd/data/images/%s' % file_name))
        print('    image_id: %s' % image_id)
        print('    question: %s' % qid2ques[qid])
        print('    ans: %s' % ans)

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
        split = 'valid'
    check(split)

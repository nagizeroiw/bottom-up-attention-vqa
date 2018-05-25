import json
import sys
import os
import random

data_root = './data/'

questions_file = {
    'train': 'v2_OpenEnded_mscoco_train2014_questions.json',
    'valid': 'v2_OpenEnded_mscoco_val2014_questions.json',
    'test': 'v2_OpenEnded_mscoco_test2015_questions.json'
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
    'valid': 'val2014/COCO_val2014_000000',
    'test': 'test2015/COCO_test2015_000000'
}


def check(split):
    questions = json.load(open(os.path.join(data_root, questions_file[split])))['questions']
    if split != 'test':
        annotations = json.load(open(os.path.join(data_root, annotations_file[split])))['annotations']

    qid2ques = {}
    for q in questions:
        qid = q['question_id']
        image_id = q['image_id']
        ques = q['question']
        qid2ques[qid] = (image_id, ques)

    if split != 'test':
        qid2ans = {}
        for a in annotations:
            qid = a['question_id']
            image_id = a['image_id']
            ans = a['multiple_choice_answer']
            qid2ans[qid] = (image_id, ans)

    while True:
        qid = raw_input('> ')
        qid = int(qid)
        if qid not in qid2ques:
            print('! cannot find question id: %s' % qid)
            continue
        image_id, ques = qid2ques[qid]

        file_name = image_path[split] + '%06d.jpg' % int(image_id)
        print('file name:', file_name)
        os.system('scp %s ./vis/' % ('jungpu6:~/vqa-butd/data/images/%s' % file_name))
        print('    image_id: %s' % image_id)
        print('    question: %s' % ques)
        if split != 'test':
            print('    ans: %s' % qid2ans[qid][1])


def count(split):
    pairs = json.load(open(os.path.join(data_root, complementary_pairs_file[split])))
    questions = json.load(open(os.path.join(data_root, questions_file[split])))['questions']
    annotations = json.load(open(os.path.join(data_root, annotations_file[split])))['annotations']

    qid2ques = {}
    for q in questions:
        qid = q['question_id']
        image_id = q['image_id']
        ques = q['question']
        qid2ques[qid] = (image_id, ques)

    if split != 'test':
        qid2ans = {}
        for a in annotations:
            qid = a['question_id']
            image_id = a['image_id']
            ans = a['multiple_choice_answer']
            qid2ans[qid] = (image_id, ans)

    clean_pairs = []
    clean_pairs_filename = 'clean_pairs_%s.json' % split

    print('counting...')
    count = 0
    for p1, p2 in pairs:
        ans1 = qid2ans[p1][1]
        ans2 = qid2ans[p2][1]
        ques1 = qid2ques[p1][1]
        ques2 = qid2ques[p2][1]
        try:
            assert ques1 == ques2
        except:
            print('! different same question: %d %d' % (p1, p2))
        
        if ans1 == ans2:
            count += 1
        else:
            clean_pairs.append((p1, p2))

    print('same answer: %d' % count)
    print('ratio: %.4f' % (1.0 * count / len(pairs)))
    print('clean pairs length: %d' % len(clean_pairs))
    json.dump(clean_pairs, open(clean_pairs_filename, 'w'))


def process(split):
    questions = json.load(open(os.path.join(data_root, questions_file[split])))['questions']
    annotations = json.load(open(os.path.join(data_root, annotations_file[split])))['annotations']
    pairs = json.load(open(os.path.join(data_root, complementary_pairs_file[split])))

    while True:

        print('> from spilt %s:' % split)

        pair = random.choice(pairs)
        q_id_1, q_id_2 = pair
        '''
        q_id_1 = int(raw_input('> '))
        print('> finding pair containing %d...' % q_id_1)
        for pair in pairs:
            if q_id_1 in pair:
                q_id_1, q_id_2 = pair
                break
        '''

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

        file_name1 = image_path[split] + '%06d.jpg' % int(i_id_1)
        print('file name 1:', file_name1)
        os.system('scp %s ./vis/' % ('jungpu6:~/vqa-butd/data/images/%s' % file_name1))
        print('> question 1: %s' % q_str_1)
        print('> answer 1: %s' % ans_1)

        file_name2 = image_path[split] + '%06d.jpg' % int(i_id_2)
        print('file name 2:', file_name2)
        os.system('scp %s ./vis/' % ('jungpu6:~/vqa-butd/data/images/%s' % file_name2))
        print('> question 2: %s' % q_str_2)
        print('> answer 2: %s' % ans_2)

        raw_input()


if __name__ == '__main__':
    try:
        split = sys.argv[1]
    except IndexError:
        split = 'valid'
    check(split)

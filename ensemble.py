from __future__ import print_function
import json
import os
from collections import Counter


json_file_root = './results'

ids = [
    'ens_tv_2345',
    'ens_tv_1206',
    'ens_tv_1111',
    'ens_tv_8989',
    'ens_tv_9090',
    'ens_tv_7853',
]
def Most_Common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

if __name__ == '__main__':

    json_files = [os.path.join(json_file_root, jid + '.json') for jid in ids]

    qid2answers = {}

    for json_file in json_files:
        with open(json_file, 'r') as fp:
            this_json = json.load(fp)
            for pair in this_json:
                qid = pair['question_id']
                ans = pair['answer']
                if qid not in qid2answers:
                    qid2answers[qid] = []
                qid2answers[qid].append(ans)

    print('questions: %d' % len(qid2answers))
    i = 0
    for q, a in qid2answers.iteritems():
        i += 1
        if i % 10000 == 0:
            print(a)
        a = Most_Common(a)
        if i % 10000 == 0:
            print(a)
        qid2answers[q] = a

    print(qid2answers.items()[:5])

    output_list = []
    for q, a in qid2answers.iteritems():
        output_list.append({
            'question_id': q,
            'answer': a
            })
    with open('results/final_results.json', 'w') as fp:
        json.dump(output_list, fp)

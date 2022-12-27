import json
import random
import os
from copy import deepcopy
input_dir = './data/diverse_data'
input_file = 'diverseqa.json'

with open(os.path.join(input_dir, input_file), 'r') as f:
    data = json.load(f)['data']

qids = []
qid2para = {}
for p_idx, item in enumerate(data):
    p = item['paragraphs'][0]
    for qa in p['qas']:
        qids.append(qa['id'])
        qid2para[qa['id']] = p_idx
data_size = len(qids)

uqa_main_size = data_size - 6 * 100000
print('data_size: [{}], uqa_main_size: [{}]'.format(data_size, uqa_main_size))
random.shuffle(qids)

output_sizes = [uqa_main_size, 100000, 100000, 100000, 100000, 100000, 100000]
output_files = ['uqa_train_main.json', 'uqa_train_refine_0.json', 'uqa_train_refine_1.json',
                'uqa_train_refine_2.json', 'uqa_train_refine_3.json', 'uqa_train_refine_4.json',
                'uqa_train_refine_5.json']

s_point = 0
for output_size, output_file in zip(output_sizes, output_files):
    output_path = os.path.join(input_dir, output_file)
    output_qids = set(qids[s_point: s_point + output_size])
    output_data = deepcopy(data)
    for item in output_data:
        p = item['paragraphs'][0]
        qas = []
        for qa in p['qas']:
            if qa['id'] in output_qids:
                qas.append(qa)
        p['qas'] = qas
    print('write to [{}]'.format(output_path))
    print(len(output_data))
    with open(output_path, 'w') as f:
        json.dump({'data': output_data}, f, indent=2, ensure_ascii=False)

    s_point += output_size

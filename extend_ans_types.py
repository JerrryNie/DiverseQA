import json
import os
from tqdm import tqdm
import benepar
import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)


parser = benepar.Parser("benepar_en3")
entity_types = ['PLACEHOLDER', 'PERSONNORPORG', 'PLACE', 'THING', 'TEMPORAL', 'NUMERIC']
span_types = ['NE', 'NP', 'ADJP', 'VP', 'S']
diverse_dir = './data/diverse_data'
raws = {}
uid_sumidx2ans_range = {}
for span_type in span_types:
    logging.info('Logging span {}'.format(span_type))
    with open(os.path.join(diverse_dir, 'cloze_clause_wikiref_data_{}.json'.format(span_type)),
              'r', encoding='utf-8') as f:
        raws[span_type] = json.load(f)
    uid_sumidx2ans_range[span_type] = {}
logging.info('Logging wikiref...')
with open('./data/diverse_data/wikiref.json', 'r', encoding='utf-8') as f:
    wikiref = json.load(f)
uid2wikiref_data = {}
if not os.path.exists(os.path.join(diverse_dir, 'uid2wikiref_data.json')):
    logging.info('Process wikiref...')
    for item in tqdm(wikiref, desc='process wikiref'):
        uid = item['uid']
        for sum_idx, sum_item in enumerate(item['summary']):
            tokens = parser.parse(sum_item).leaves()
            sum_item = ' '.join(tokens)
            item['summary'][sum_idx] = sum_item
        uid2wikiref_data[uid] = {
            'document': item['document'],
            'summary': item['summary']
        }
    with open(os.path.join(diverse_dir, 'uid2wikiref_data.json'), 'w', encoding='utf-8') as f:
        json.dump(uid2wikiref_data, f, indent=4)
else:
    logging.info('uid2wikiref_data exists! Logging...')
    with open(os.path.join(diverse_dir, 'uid2wikiref_data.json'), 'r', encoding='utf-8') as f:
        uid2wikiref_data = json.load(f)
for span_type in span_types:
    for passage in tqdm(raws[span_type]['data'], desc='{} preprocess...'.format(span_type)):
        for p in passage['paragraphs']:
            context = p['context']
            for qa in p['qas']:
                question = qa['question']
                question = question.replace('(', '-LRB-').replace(')', '-RRB-').replace('[', '-LSB-')
                question = question.replace(']', '-RSB-').replace('{', '-LCB-').replace('}', '-RCB-')
                qid = qa['id']
                uid = qid.split('_')[0]
                question_first_part = None
                question_second_part = None
                for entity_type in entity_types:
                    if len(question.split(entity_type)) > 1:
                        question_first_part = question.split(entity_type)[0]
                        question_second_part = question.split(entity_type)[1]
                        break
                assert question_first_part is not None and question_second_part is not None, 'q: {}, first: {}, second: {}'.format(
                    question, question_first_part, question_second_part
                )
                summary = uid2wikiref_data[uid]['summary']
                summary = [s.replace('(', '-LRB-') for s in summary]
                summary = [s.replace(')', '-RRB-') for s in summary]
                summary = [s.replace('[', '-LSB-') for s in summary]
                summary = [s.replace(']', '-RSB-') for s in summary]
                summary = [s.replace('{', '-LCB-') for s in summary]
                summary = [s.replace('}', '-RCB-') for s in summary]
                summary_idx = -1
                for sum_idx, summary_item in enumerate(summary):
                    if len(question_first_part) >= 1 and summary_item.find(question_first_part) != -1:
                        summary_idx = sum_idx
                        break
                    if len(question_second_part) >= 1 and summary_item.find(question_second_part) != -1:
                        summary_idx = sum_idx
                        break
                if summary_idx == -1 and len(question_first_part) == 0 and len(question_second_part) == 0:
                    pass
                else:
                    assert summary_idx != -1, 'qa: {}, summary: {}, first: ***{}***,***{}***; second: ****{}****,****{}****'.format(qa,
                                                                                                        summary,
                                                                                                        question_first_part,
                                                                                                        question_second_part,
                                                                                                        summary[0].find(question_first_part),
                                                                                                        summary[0].find(question_second_part))
                qa['summary_idx'] = summary_idx
                answer = qa['answers'][0]
                answer_start = answer['answer_start']
                answer_text = answer['text']
                answer_end = answer_start + len(answer_text)  # [answer_start, answer_end)
                if uid not in uid_sumidx2ans_range[span_type]:
                    uid_sumidx2ans_range[span_type][uid] = {}
                if summary_idx not in uid_sumidx2ans_range[span_type][uid]:
                    uid_sumidx2ans_range[span_type][uid][summary_idx] = []
                assert answer_text == context[answer_start: answer_end]
                uid_sumidx2ans_range[span_type][uid][summary_idx].append({
                    'question': question,
                    'text': answer_text,
                    'answer_start': answer_start,
                    'answer_end': answer_end
                })

for passage in tqdm(raws['NE']['data'], desc='extend NE answers...'):
    for p in passage['paragraphs']:
        context = p['context']
        for qa in p['qas']:
            qid = qa['id']
            uid = qid.split('_')[0]
            summary_idx = qa['summary_idx']
            qa['answers'][0]['span_type'] = 'NE'
            for entity_type in entity_types:
                if entity_type != 'PLACEHOLDER' and entity_type in qa['question']:
                    qa['answers'][0]['entity_type'] = entity_type
            if summary_idx == -1:
                continue
            answer_start = qa['answers'][0]['answer_start']
            answer_end = answer_start + len(qa['answers'][0]['text'])
            for span_type in span_types:
                if span_type == 'NE':
                    continue
                if uid not in uid_sumidx2ans_range[span_type]:
                    continue
                if summary_idx not in uid_sumidx2ans_range[span_type][uid]:
                    continue
                answer_ranges = uid_sumidx2ans_range[span_type][uid][summary_idx]
                for answer_range in answer_ranges:
                    if (answer_range['answer_start'] < answer_start and answer_range['answer_end'] >= answer_end
                            or answer_range['answer_start'] <= answer_start and answer_range['answer_end'] > answer_end):
                        clause = qa['question'] + ' ' + qa['answers'][0]['text']
                        clause_len = len(clause.split()) - 1  # ignore the placeholder
                        candidate_ans_len = len(answer_range['text'].split())
                        if clause_len * 0.8 < candidate_ans_len:
                            continue
                        qa['answers'][0]['span_type'] = span_type
                        qa['answers'][0]['answer_start'] = answer_range['answer_start']
                        qa['answers'][0]['text'] = answer_range['text']
                        qa['question'] = answer_range['question'].replace('PLACEHOLDER',
                                                                          qa['answers'][0]['entity_type'])
                        answer_start = answer_range['answer_start']
                        answer_end = answer_range['answer_end']
                        assert answer_range['text'] == context[answer_start: answer_end]

with open(os.path.join(diverse_dir, 'cloze_clause_wikiref_data_diverse_answer_span_80.json'), 'w', encoding='utf-8') as f:
    json.dump(raws['NE'], f, indent=4)

raw = raws['NE']

span2cnt = {}
for passage in tqdm(raw['data'], desc='span stat'):
    for p in passage['paragraphs']:
        for qa in p['qas']:
            ans_type = qa['answers'][0]['span_type']
            if ans_type not in span2cnt:
                span2cnt[ans_type] = 1
            else:
                span2cnt[ans_type] += 1
logging.info(span2cnt)

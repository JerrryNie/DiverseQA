# -*- coding: utf-8 -*-
import sys
import os
import json
import time
import numpy as np
from tqdm import tqdm
import random
import logging
from collections import Counter
import spacy
import copy
import benepar
import nltk
from data_utils import reformulate_quesiton, filter_data_given_qids_v2
from cloze2natural import identity_translate
from wikiref_process import get_clause_v2
import argparse
import pickle
import torch

entity_category = {
    'PERSONNORPORG': "PERSON, NORP, ORG".replace(' ', '').split(','),
    'PLACE': "GPE, LOC, FAC".replace(' ', '').split(','),
    'THING': 'PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE'.replace(' ', '').split(','),
    'TEMPORAL': 'TIME, DATE'.replace(' ', '').split(','),
    'NUMERIC': 'PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL'.replace(' ', '').split(',')
}
entity_type_map = {}
for cate in entity_category:
    for item in entity_category[cate]:
        entity_type_map[item] = cate

spacy_ner = spacy.load("en_core_web_sm", disable=['parser', 'tagger'])
spacy_tagger = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
spacy_parser = spacy.load("en_core_web_sm", disable=['ner', 'tagger'])
bene_parser = benepar.Parser("benepar_en3")


def get_new_question(context, doc, answer, answer_start, summary, qtype):
    sentences = []
    for sent in summary:
        if answer in sent:
            sentences.append(sent)
    if len(sentences) == 0 or answer_start == -1:
        return None

    context_sent = None
    char_cnt = 0
    for sent_item in doc.sents:
        sent = sent_item.text
        if char_cnt <= answer_start < char_cnt + len(sent):
            context_sent = sent
            break
        else:
            char_cnt += len(sent)
            while char_cnt < len(context) and context[char_cnt] == ' ':
                char_cnt += 1

    if context_sent is None:
        return None

    c_tokens = []
    c_doc = spacy_tagger(context_sent)
    for token in c_doc:
        if not token.is_stop:
            c_tokens.append(token.lemma_)

    result = []
    for sent in sentences:
        sent_doc = spacy_tagger(sent)
        score = 0
        for token in sent_doc:
            if token.is_stop:
                continue
            if token.lemma_ in c_tokens:
                score += 1
        result.append([score, sent])
    result = sorted(result, key=lambda x: x[0])
    sentence = result[-1][1]
    cloze_text = None
    for clause in get_clause_v2(sentence, bene_parser):
        if answer in clause:
            cloze_text = clause.replace(answer, qtype, 1)
            break
    if cloze_text is None:
        return None

    new_question = identity_translate(reformulate_quesiton(cloze_text, spacy_parser, reform_version=1))
    if new_question.startswith('Wh') or new_question.startswith('How'):
        return new_question
    else:
        return None


def get_answer_start(context, answer, orig_doc_start):
    begin_index = len(' '.join(context.split(' ')[:orig_doc_start]))
    answer_index = context.find(answer, begin_index)
    return answer_index


def generate(args, input_file, nbest_file, output_file, hard_em=False, score_lower_bound=0.5, debug=False):
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]
    with open(nbest_file, "r", encoding='utf-8') as reader:
        nbest_data = json.load(reader)

    q_count = 0
    filter_qids = []

    new_data = []
    for entry in tqdm(copy.deepcopy(input_data), desc='filtering...'):
        for paragraph in entry['paragraphs']:
            for qa in paragraph['qas']:
                answer_text = qa['answers'][0]['text']
                qid = qa['id']
                answers = [ans['text'] for ans in nbest_data[qid]]
                if answer_text in answers[0: args.top]:
                    filter_qids.append(qid)

    new_data_substring = []
    substring_qa_cnt = 0
    for entry in tqdm(copy.deepcopy(input_data), desc='filter substring...'):
        paras = []
        for paragraph in entry['paragraphs']:
            qas_substring = []
            for qa in paragraph['qas']:
                answer_text = qa['answers'][0]['text']
                if 'span_type' in qa['answers'][0]:
                    span_type = qa['answers'][0]['span_type']
                else:
                    span_type = 'NE'
                qid = qa['id']
                cnt = 0
                for ans in (nbest_data[qid][:1] if hard_em else nbest_data[qid]):
                    if ans['probability'] < score_lower_bound:
                        continue
                    new_qa = copy.deepcopy(qa)
                    new_qa['id'] = qa['id'] + '_%d' % cnt
                    ans['text'] = ans['text'].strip()

                    if (answer_text == ans['text']) or (ans['text'] in answer_text):
                        if answer_text == ans['text']:
                            pass
                        elif span_type == 'NE':
                            qas_substring.append(new_qa)
                        if (new_qa['question'] is None) or (new_qa['answers'][0]['answer_start'] == -1):
                            continue
                    cnt += 1

            if len(qas_substring) > 0:
                substring_qa_cnt += len(qas_substring)
                paragraph['qas'] = qas_substring
                paras.append(paragraph)
        if len(paras) == 0:
            continue
        entry['paragraphs'] = paras
        new_data_substring.append(entry)

    random.shuffle(filter_qids)
    filter_data = filter_data_given_qids_v2(input_data, filter_qids, args.use_unanswerable_instances)
    new_data = filter_data + new_data_substring
    q_count = len(filter_qids)
    print('New Questions(V3)', q_count + substring_qa_cnt)
    print('Filter {} qas; substring {} qas'.format(q_count, substring_qa_cnt))
    json.dump({"version": "v2.0", 'data': new_data}, open(output_file, 'w', encoding='utf-8'))
    print('output_file_path: {}'.format(output_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, type=str,
                        help="output_file")
    parser.add_argument("--input", default=None, type=str,
                        help="input_file")
    parser.add_argument("--nbest", default=None, type=str,
                        help="nbest_file")
    parser.add_argument("--score_threshold", default=0.3, type=float,
                        help="The threshold of generating new qa data.")
    parser.add_argument("--nonsubstring_score_threshold", default=0.3, type=float,
                        help="The threshold of generating new qa data.")
    parser.add_argument("--seed", default=42, type=int,
                        help="random seed")
    parser.add_argument("--top", default=1, type=int)
    args = parser.parse_args()
    random.seed(args.seed)
    assert not (args.only_filter_data and args.only_refine_data)
    device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    tensor0 = torch.randn(40000, 60000, device=device0)
    tensor1 = torch.randn(40000, 60000, device=device1)
    tensor2 = torch.randn(40000, 60000, device=device2)
    generate(args, args.input, args.nbest, args.output, score_lower_bound=args.score_threshold)


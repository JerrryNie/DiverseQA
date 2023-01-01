# -*- coding: utf-8 -*-
import sys, os, json, time
from tqdm import tqdm
import spacy
import benepar
import argparse
import nltk
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
data_dir = './data/diverse_data'

entity_category = {
    'PERSONNORPORG' : "PERSON, NORP, ORG".replace(' ','').split(','),
    'PLACE' : "GPE, LOC, FAC".replace(' ','').split(','),
    'THING' : 'PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE'.replace(' ','').split(','),
    'TEMPORAL': 'TIME, DATE'.replace(' ','').split(','),
    'NUMERIC' : 'PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL'.replace(' ','').split(',')
}
entity_type_map = {}
for cate in entity_category:
    for item in entity_category[cate]:
        entity_type_map[item] = cate


def search_span_from_tree(tree_node, span_type):
    """Search specific type of spans from a constituency tree (in a "bfs" way)

        Parameters:
            tree_node: the node of a constituency (sub-)tree
            span_type: the type of span to be extracted

    """
    spans = []
    if isinstance(tree_node, nltk.Tree):
        for i in range(len(tree_node)):
            spans += search_span_from_tree(tree_node[i], span_type)
        if tree_node.label() == span_type:
            spans.append(tree_node)
    return spans


def search_sbar_from_tree(tree_node):
    if not isinstance(tree_node, nltk.Tree):
        return []
    clause = []
    for i in range(len(tree_node)):
        clause += search_sbar_from_tree(tree_node[i])
    if tree_node.label() == 'S' or tree_node.label() == 'SBAR':
        clause.append(tree_node)
    return clause


def get_clause_v2(sentence, predictor):
    parsing_tree = predictor.parse(sentence)
    sbar = search_sbar_from_tree(parsing_tree)[:-1]
    result = []
    for node in sbar:
        if node.label() == 'S':
            item = ' '.join(node.leaves())
            if len(item.split()) <= 5:
                continue
            result.append(item)

    result = sorted(result, key=lambda x: len(x))
    result2 = []
    sentence = ' '.join(parsing_tree.leaves())
    clauses = sentence.split(',')
    for i in range(len(clauses)):
        item, p = clauses[i], i+1
        while len(item.split()) < 10 and p < len(clauses):
            item = ','.join([item, clauses[p]])
            p += 1
        result2.append(item.strip())

    result2 = sorted(result2, key=lambda x: len(x))
    return result + result2


def get_answer_start(answer, question, sentences, tagger):
    q_tokens = []
    q_doc = tagger(question)
    for token in q_doc:
        if not token.is_stop:
            q_tokens.append(token.lemma_)

    result = []
    for sent in sentences:
        if sent.find(answer) == -1:
            continue
        sent_doc = tagger(sent)
        score = 0
        for token in sent_doc:
            if token.is_stop:
                continue
            if token.lemma_ in q_tokens:
                score += 1
        result.append([score, sent])
    if len(result) == 0:
        return -1
    else:
        result = sorted(result, key=lambda x: x[0])
        res_sent = result[-1][1]
        answer_start = ' '.join(sentences).find(res_sent) + res_sent.find(answer)
        return answer_start


def search_answer_span_via_type(tree_node, span_type):
    """search span via span_type from a sentence(parsed as a tree_node)

    Args:
        tree_node (nltk.Tree): a parsed sentence
        span_type (str): the type of the span to be extracted

    Returns:
        list: a list of the satisfying spans having the span_type
    """

    if len(tree_node._.labels) == 0:
        return []
    spans = []
    children = list(tree_node._.children)
    for child in children:
        spans += search_answer_span_via_type(child, span_type)
    if tree_node._.labels[0] == span_type:
        spans.append(str(tree_node))
    return spans


def get_answer_by_type(sentence, span_type):
    """Extract specific type of spans in the given sentence.
        We use Benepar constituency parser to complete it.
        The root sentence is ignored in the span extracting process.
        Possible span type: S (clause), VP (verb phrase), ADJP (adjective phrase)

        Parameters:
            sentence: String
                a sentence to be parsed and extract answer from
            type: String
                the type of span to be extracted
                avalible values: 'S', 'VP', 'ADJP'
        Returns:
            a list of spans
    """
    try:
        doc = nlp(sentence)
        sents = list(doc.sents)
        sent = sents[0]
    except Exception as e:
        print('Anomal sentence: ' + sentence)
        print('Error:' + str(e))
        return None
    spans = search_answer_span_via_type(tree_node=sent, span_type=span_type)
    return spans


def get_cloze_data_v2(input_data, span_type):
    """Extract specific type of spans from the text

    Args:
        input_data (list): a list of documents (with its own cited document) crawled from the Wikipedia
        span_type (str): a string may be 'VP', 'ADJP' or 'S'
    Returns:
        dict: contains the cloze style training data
    """
    parser = benepar.Parser("benepar_en3")

    tagger = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

    cloze_data = []

    q_count = 0
    c_count = 0

    for item in tqdm(input_data, desc="cloze"):
        entry = {}
        entry['title'] = item["document"][0]
        paragraph = {}
        paragraph["context"] = ' '.join(item["document"])

        qas = []

        for sent_idx, sent in enumerate(item['summary']):
            spans = get_answer_by_type(sentence=sent, span_type=span_type)
            if spans is None:
                continue
            try:
                clause = get_clause_v2(sent, parser)
            except Exception as e:
                continue

            for ent in spans:
                answer = ent.strip()
                question = None
                for each in clause:
                    if len(answer.split()) >= len(each.split()):
                        continue
                    if each.find(answer) != -1:
                        question = each.replace(answer, 'PLACEHOLDER', 1)
                        break
                if not question:
                    continue

                answer_start = get_answer_start(answer, question, item['document'], tagger)
                if answer_start == -1:
                    continue

                qas.append({
                    "question": question,
                    "id": "%s_%d" % (item['uid'], q_count),
                    "is_impossible": False,
                    "answers": [
                        {
                            "answer_start": answer_start,
                            "text": answer,
                            "type": "",
                            "sent_idx": sent_idx
                        }
                    ],
                    "plausible_answers": []
                })
                q_count += 1

        paragraph['qas'] = qas
        paragraph['summary'] = item['summary']
        entry['paragraphs'] = [paragraph]

        cloze_data.append(entry)
        c_count += 1

    print('Questions Number', q_count)    
    return {"version": "v2.0", 'data': cloze_data}


def get_cloze_data(input_data):
    parser = benepar.Parser("benepar_en3")

    ner = spacy.load("en_core_web_sm", disable=['parser', 'tagger'])
    tagger = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

    cloze_data = []

    q_count = 0
    c_count = 0

    for item in tqdm(input_data, desc="cloze"):
        entry = {}
        entry['title'] = item["document"][0]
        paragraph = {}
        paragraph["context"] = ' '.join(item["document"])

        qas = []

        for sent in item['summary']:
            sent_doc = ner(sent)
            try:
                clause = get_clause_v2(sent, parser)
            except Exception as e:
                continue
            
            for ent in sent_doc.ents:
                answer = ent.text

                question = None
                for each in clause:
                    if each.find(answer) != -1:
                        question = each.replace(answer, entity_type_map[ent.label_], 1)
                        break
                else:
                    question = sent[:ent.start_char] + \
                            sent[ent.start_char:].replace(answer,entity_type_map[ent.label_], 1)
                if not question:
                    continue

                answer_start = get_answer_start(answer, question, item['document'], tagger)
                if answer_start == -1:
                    continue

                qas.append({
                            "question": question,
                            "id": "%s_%d"%(item['uid'], q_count) ,
                            "is_impossible": False,
                            "answers": [
                                {
                                "answer_start": answer_start,
                                "text": answer,
                                "type": ent.label_
                                }
                            ],
                            "plausible_answers": []
                })
                q_count += 1

        paragraph['qas'] = qas
        entry['paragraphs'] = [paragraph]
        
        cloze_data.append(entry)
        c_count += 1

    print('Questions Number', q_count)    
    return {"version": "v2.0", 'data': cloze_data}


def raw_to_spans(args, input_data, span_type):
    print('span_type: {}'.format(span_type))
    if span_type == 'NE':
        cloze_clause_data = get_cloze_data(input_data)
    else:
        cloze_clause_data = get_cloze_data_v2(input_data, span_type=span_type)
    json.dump(cloze_clause_data,
              open(os.path.join(args.output_dir, 'cloze_clause_wikiref_data_{}.json'.format(span_type)),
                   "w",
                   encoding='utf-8'),
              indent=4)


def main(args):
    input_file = os.path.join(args.input_dir, args.input_file)
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)
    span_types = [args.span_type]
    for span_type in span_types:
        raw_to_spans(args, input_data=input_data, span_type=span_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default='./data/diverse_data', type=str)
    parser.add_argument("--input_file", default="wikiref.json", type=str)
    parser.add_argument("--output_dir", default='./data/diverse_data', type=str)
    parser.add_argument("--span_type", default="span type", choices=['NE', 'NP', 'ADJP', 'VP', 'S'],
                        type=str)
    args = parser.parse_args()
    assert os.path.exists(os.path.join(args.input_dir, args.input_file))
    assert os.path.exists(args.output_dir)
    main(args)

# -*- coding: utf-8 -*-
import pickle
import os
import sys
import json
import time
import numpy as np
from tqdm import tqdm
import random
import logging
import argparse
import subprocess
import glob
from pytorch_transformers import WEIGHTS_NAME

from evaluate import evaluate

logger = logging.getLogger('multi-turn')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)


def get_nbest_file(model_dir, dev_file, params):
    if params.base:
        command = 'python run_squad.py --model_type bert \
                --model_name_or_path %s --do_eval --train_file %s \
                --predict_file %s --max_seq_length 384  --doc_stride 128 --output_dir %s \
                --per_gpu_eval_batch_size=128 --eval_prefix dev\
                --bert_model %s --n_best_size %s' % (params.model_name_or_path,
                                                     params.predict_file,
                                                     dev_file, model_dir, params.model_name_or_path,
                                                     params.n_best_size)
        if params.model_name_or_path.endswith('uncased'):
            command += ' --do_lower_case'
    else:
        command = 'python run_squad_swep.py --model_type bert \
                --model_name_or_path %s --do_eval --do_lower_case --train_file %s \
                --bert_model %s \
                --predict_file %s --max_seq_length 384  --doc_stride 128 --output_dir %s\
                --n_best_size %s\
                --per_gpu_eval_batch_size=128 --hidden_size 1024 --eval_prefix dev' % (params.model_name_or_path,
                                                                                       params.predict_file,
                                                                                       params.model_name_or_path,
                                                                                       dev_file,
                                                                                       model_dir,
                                                                                       params.n_best_size)
    if params.fp16:
        command += ' --fp16'
    nbest_file = os.path.join(model_dir, 'nbest_predictions_dev.json')
    if params.debug and os.path.isfile(nbest_file):
        logger.info('%s already existed and we use it.' % nbest_file)
    else:
        logger.info('Generating nbest file...')
        subprocess.Popen(command, shell=True).wait()

    if not os.path.isfile(nbest_file):
        logger.error('Nbest file %s is not found.' % nbest_file)
        exit()
    logger.info('Got nbest file %s' % nbest_file)
    return nbest_file


def get_new_train_file(dev_file, nbest_file, model_dir, params, output_dir):
    new_train_file = os.path.join(output_dir, 'train_data_for_next_turn.json')
    command = 'python generate_new_qadata.py --input %s --nbest %s --output %s --top %d \
            --score_threshold %.4f --nonsubstring_score_threshold %.4f \
            --seed %d' % (dev_file, nbest_file,
                          new_train_file,
                          params.top,
                          params.score_threshold,
                          params.nonsubstring_score_threshold,
                          params.seed)
    if params.substring:
        command += ' --substring'
    if params.nonsubstring:
        command += ' --nonsubstring'
    subprocess.Popen(command, shell=True).wait()
    if not os.path.isfile(new_train_file):
        logger.error('New train file %s is not found.' % new_train_file)
        exit()
    logger.info('Got new train file %s' % new_train_file)
    return new_train_file


def do_evaluate(dataset_file, prediction_file):
    with open(dataset_file) as df:
        dataset_json = json.load(df)
        dataset = dataset_json['data']
    with open(prediction_file) as pf:
        predictions = json.load(pf)
    return evaluate(dataset, predictions)


def train_model(train_file, model_dir, output_dir, params):
    if params.base:
        command = 'python -m torch.distributed.launch --nproc_per_node=3 --master_port %s run_squad.py \
            --model_type bert  --model_name_or_path %s --do_train  --do_eval\
            --train_file %s --predict_file %s \
            --learning_rate %s  --num_train_epochs 1.0  --max_seq_length 384  --doc_stride 128 \
            --output_dir %s  --per_gpu_eval_batch_size=128  --per_gpu_train_batch_size=4 --seed %d\
            --gradient_accumulation_steps 2\
            --logging_steps 1000  --save_steps 1000 --eval_all_checkpoints\
            --overwrite_output_dir --overwrite_cache\
            --bert_model %s --init_model_dir %s' % (args.master_port, model_dir, train_file, params.predict_file,
                                                    params.lr, output_dir,
                                                    params.seed, params.model_name_or_path, model_dir)
        if params.model_name_or_path.endswith('uncased'):
            command += ' --do_lower_case'
    else:
        command = 'python -m torch.distributed.launch --nproc_per_node=3 --master_port %s run_squad_swep.py \
            --model_type bert  --model_name_or_path %s --do_train  --do_eval  --do_lower_case \
            --train_file %s --predict_file %s \
            --learning_rate %s  --num_train_epochs 1.0  --max_seq_length 384  --doc_stride 128 \
            --output_dir %s  --per_gpu_eval_batch_size=128  --per_gpu_train_batch_size=4 --seed %d\
            --gradient_accumulation_steps 2\
            --logging_steps 1000  --save_steps 1000 --eval_all_checkpoints\
            --overwrite_output_dir --overwrite_cache --hidden_size 1024\
            --bert_model %s --init_model_dir %s' % (args.master_port,
                                                    model_dir,
                                                    train_file,
                                                    params.predict_file,
                                                    params.lr,
                                                    output_dir,
                                                    params.seed,
                                                    params.model_name_or_path,
                                                    model_dir)
    if params.fp16:
        command += ' --fp16'
    if params.warmup_steps is not None:
        command += ' --warmup_steps {}'.format(params.warmup_steps)
    subprocess.Popen(command, shell=True).wait()

    # select best model for next turn
    new_model_dir = output_dir
    score = do_evaluate(params.predict_file, os.path.join(output_dir, 'predictions_.json'))[params.metric]

    for filename in os.listdir(output_dir):
        if (not filename.startswith('predictions_')) or (filename == 'predictions_.json'):
            continue
        new_score = do_evaluate(params.predict_file, os.path.join(output_dir, filename))[params.metric]
        if new_score > score:
            score = new_score
            ckpt = filename.replace('.json', '').replace('predictions_', 'checkpoint-')
            new_model_dir = os.path.join(output_dir, ckpt)
            subprocess.Popen('cp %s/vocab.txt  %s' % (output_dir, new_model_dir), shell=True).wait()
            subprocess.Popen('cp %s/special_tokens_map.json  %s' % (output_dir, new_model_dir), shell=True).wait()
            subprocess.Popen('cp %s/added_tokens.json  %s' % (output_dir, new_model_dir), shell=True).wait()
            subprocess.Popen('cp %s/%s  %s/predictions_.json' % (output_dir, filename, new_model_dir), shell=True).wait()

    return new_model_dir, score


def main(params):
    dev_data_name = os.path.join(args.refine_data_dir, 'uqa_train_refine_%d.json')

    model_dir = os.path.join(params.output_dir, 'init')
    if not os.path.exists(model_dir):
        subprocess.Popen('mkdir -p %s' % model_dir, shell=True).wait()
    logger.info('Copy model from %s to %s.' % (params.model_dir, model_dir))
    subprocess.Popen('cp %s/vocab.txt %s' % (params.model_dir, model_dir), shell=True).wait()
    subprocess.Popen('cp %s/special_tokens_map.json %s' % (params.model_dir, model_dir), shell=True).wait()
    subprocess.Popen('cp %s/added_tokens.json %s' % (params.model_dir, model_dir), shell=True).wait()
    subprocess.Popen('cp %s/predictions_.json %s' % (params.model_dir, model_dir), shell=True).wait()
    subprocess.Popen('cp %s/bert_base.pt %s' % (params.model_dir, model_dir), shell=True).wait()
    subprocess.Popen('cp %s/tokenizer_config.json %s' % (params.model_dir, model_dir), shell=True).wait()

    if os.path.exists(os.path.join(model_dir, 'predictions_.json')):
        current_score = do_evaluate(params.predict_file, os.path.join(model_dir, 'predictions_.json'))[params.metric]
    else:
        current_score = 0.0

    order = [0, 1, 2, 3, 4, 5]
    break_point = {}
    break_point_path = os.path.join(params.output_dir, 'break_point.json')
    if params.load_from_break_point:
        if not os.path.exists(break_point_path):
            params.load_from_break_point = False
        else:
            with open(break_point_path, 'r', encoding='utf-8') as f:
                break_point = json.load(f)
    for step, idx in enumerate(order):
        if params.load_from_break_point and step < break_point['step']:
            continue
        if params.load_from_break_point and step == break_point['step']:
            current_score = break_point['current_score']
            model_dir = break_point['model_dir']
        else:
            break_point['step'] = step
            break_point['current_score'] = current_score
            break_point['model_dir'] = model_dir
            with open(os.path.join(params.output_dir, 'break_point.json'), 'w', encoding='utf-8') as f:
                json.dump(break_point, f, indent=4)
        logger.info('-' * 80)
        logger.info('Prepare for turn_%d / Current %s %.2f/ Current model %s' % (step, params.metric,
                                                                                 current_score, model_dir))
        dev_file = dev_data_name % idx
        output_dir = os.path.join(params.output_dir, 'turn_%d' % step)
        if not os.path.exists(output_dir):
            subprocess.Popen('mkdir -p %s' % output_dir, shell=True).wait()

        nbest_file = get_nbest_file(model_dir, dev_file, params)
        new_train_file = get_new_train_file(dev_file, nbest_file, model_dir, params, output_dir)

        new_model_dir, new_score = train_model(new_train_file, model_dir, output_dir, params)

        if new_score > current_score:
            model_dir = new_model_dir
            current_score = new_score
            logger.info('Find better model %s and %s is %.2f' % (model_dir, params.metric, current_score))

        params.score_threshold = params.score_threshold * params.threshold_rate
        params.nonsubstring_score_threshold = params.nonsubstring_score_threshold * params.nonsubstring_threshold_rate

    # record the best model and its dir
    break_point['step'] = len(order)
    break_point['current_score'] = current_score
    break_point['model_dir'] = model_dir
    with open(os.path.join(params.output_dir, 'break_point.json'), 'w', encoding='utf-8') as f:
        json.dump(break_point, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--refine_data_dir", default=None, type=str, required=True,
                        help="RefQA data for refining.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory.")
    parser.add_argument("--model_dir", default=None, type=str, required=True,
                        help="The init model directory.")
    parser.add_argument("--predict_file", default='dev-v1.1.json', type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--score_threshold", default=0.3, type=float,
                        help="The threshold of generating new qa data.")
    parser.add_argument("--threshold_rate", default=1.0, type=float,
                        help="The change rate of the threshold")
    parser.add_argument("--nonsubstring_score_threshold", default=0.3, type=float,
                        help="The threshold of generating new qa data.")
    parser.add_argument("--nonsubstring_threshold_rate", default=1.0, type=float,
                        help="The change rate of the threshold")
    parser.add_argument("--seed", default=42, type=int,
                        help="seed")
    parser.add_argument("--fp16", action='store_true',
                        help="fp16 training")
    parser.add_argument("--debug", action='store_true',
                        help="debug training")
    parser.add_argument("--base", action='store_true',
                        help='use bert-base or not')
    parser.add_argument("--next_step", default=None, type=int,
                        help='starting from a specific step')
    parser.add_argument("--master_port", default=29505, type=int)
    parser.add_argument("--task", default=None)
    parser.add_argument("--metric", default='f1', type=str)
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str)
    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str)
    parser.add_argument("--n_best_size", type=int, default=20)
    parser.add_argument("--top", default=1, type=int)
    parser.add_argument("--load_from_break_point", action='store_true')
    parser.add_argument("--substring", action='store_true')
    parser.add_argument("--nonsubstring", action='store_true')
    parser.add_argument("--warmup_steps", default=None, type=float)
    parser.add_argument("--lr", default=3e-5, type=float)
    args = parser.parse_args()
    args.output_dir = args.output_dir.replace(
        '[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', ''))
    main(args)

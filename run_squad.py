# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (Bert)."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import glob
import pickle
import ntpath
from math import ceil
import subprocess

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from models import DiverseQA

from tensorboardX import SummaryWriter

from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertTokenizer)
from transformers import BertForQuestionAnswering, AutoTokenizer

# from pytorch_transformers import AdamW, WarmupLinearSchedule
from transformers import AdamW, get_linear_schedule_with_warmup

from utils_squad import (read_squad_examples, convert_examples_to_features_parallel,
                         RawResult, write_predictions)

# The follwing import is the official SQuAD evaluation script (2.0).
# You can remove it from the dependencies if you are using this script outside of the library
# We've added it here for automated tests (see examples/test_examples.py file)
from utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def save_model(args, output_dir, model):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ckpt_file = os.path.join(output_dir, "bert_base.pt")
    ckpt = {"args": args, "state_dict": model.state_dict()}
    torch.save(ckpt, ckpt_file)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = ceil(len(train_dataloader) / args.gradient_accumulation_steps * args.num_train_epochs)

    if t_total < args.min_steps:
        t_total = args.min_steps
        args.num_train_epochs = args.min_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps * t_total, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.warning("***** Running training *****")
    logger.warning("  Num examples = %d", len(train_dataset))
    logger.warning("  Num Epochs = %d", args.num_train_epochs)
    logger.warning("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.warning("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.warning("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.warning("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':       batch[0],
                      'attention_mask':  batch[1], 
                      'token_type_ids':  batch[2],  
                      'start_positions': batch[3], 
                      'end_positions':   batch[4]}
            if args.answer_type:
                inputs.update({'answer_types': batch[7]})
            outputs = model(**inputs)
            nll, kl = outputs[0], outputs[1]
            beta = args.beta
            loss = nll + kl * beta
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step() # Update learning rate schedule
                global_step += 1

                if args.local_rank in [-1, 0]:
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('loss', tr_loss - logging_loss, global_step)
                    tb_writer.add_scalar('gradient_norm', grad_norm, global_step)
                    tb_writer.add_scalar('beta', beta, global_step)
                    if global_step % args.logging_steps == 0:
                        logger.warning('Step {}, LR {}, Loss {}'.format(global_step, scheduler.get_last_lr()[0], tr_loss - logging_loss))
                    logging_loss = tr_loss
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                        logger.warning('Eval F1 {}'.format(results['f1']))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    save_model(args, output_dir, model)
                    logger.warning("Saving model checkpoint to %s", output_dir)
                    if args.few_shot:
                        args.save_steps = args.save_steps * 2
                        if args.save_steps > args.logging_steps:
                            args.save_steps = args.logging_steps

            if args.max_steps > 0 and global_step > args.max_steps:
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_model(args, output_dir, model)
        tb_writer.close()

    return global_step, (tr_loss / global_step) if global_step else 0


def evaluate(args, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.warning("***** Running evaluation {} *****".format(prefix))
    logger.warning("  Num examples = %d", len(dataset))
    logger.warning("  Batch size = %d", args.eval_batch_size)
    all_results = []
    total_eval_steps = len(eval_dataloader)
    logger.warning('Total eval steps: {}'.format(total_eval_steps))
    for step, batch in enumerate(tqdm(eval_dataloader, desc='eval')):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2]
                      }
            example_indices = batch[3]
            outputs = model(**inputs)
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = RawResult(unique_id=unique_id,
                               start_logits=to_list(outputs[0][i]),
                               end_logits=to_list(outputs[1][i]))
            all_results.append(result)

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    write_predictions(examples, features, all_results, args.n_best_size,
                      args.max_answer_length, args.do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                      args.version_2_with_negative, args.null_score_diff_threshold)
    # Evaluate with the official SQuAD script
    evaluate_options = EVAL_OPTS(data_file=args.predict_file,
                                 pred_file=output_prediction_file,
                                 na_prob_file=output_null_log_odds_file)
    results = evaluate_on_squad(evaluate_options)
    return results


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    # Load data features from cache or dataset file
    input_file = args.predict_file if evaluate else args.train_file
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train', str(args.max_seq_length),  list(filter(None, args.model_name_or_path.split('/'))).pop(),
        list(filter(None, input_file.split('/'))).pop()))
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        logger.warning("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.warning("Creating features from dataset file at %s", input_file)
        examples = read_squad_examples(input_file=input_file,
                                       is_training=not evaluate,
                                       version_2_with_negative=args.version_2_with_negative)
        features = convert_examples_to_features_parallel(examples=examples,
                                                         tokenizer=tokenizer,
                                                         max_seq_length=args.max_seq_length,
                                                         doc_stride=args.doc_stride,
                                                         max_query_length=args.max_query_length,
                                                         is_training=not evaluate)
        if (args.local_rank in [-1, 0]) and (not evaluate):
            logger.warning("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    if evaluate:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask)
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        if args.answer_type:
            all_answer_types = torch.tensor([f.answer_type for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_start_positions, all_end_positions,
                                    all_cls_index, all_p_mask, all_answer_types)
        else:
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                    all_start_positions, all_end_positions,
                                    all_cls_index, all_p_mask)

    if output_examples:
        return dataset, examples, features
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str, required=True,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--eval_prefix", default="", type=str,
                        help="Evaluate prefix")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--version_2_with_negative', action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--min_steps", default=-1, type=int,
                        help="If > 0: ")
    parser.add_argument("--warmup_steps", default=0.0, type=float,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--few_shot', action='store_true', help='few-shot learning')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_cycle_beta", action='store_true')
    parser.add_argument("--weighted_loss", action='store_true')
    parser.add_argument("--init_model_dir", default=None, type=str,
                        help='the dir of the init model for continuing training')
    parser.add_argument('--threshold', action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--combine", action='store_true')
    parser.add_argument("--not_trained_parallel", action='store_true')
    parser.add_argument("--threshold_method", default='linear', type=str)
    parser.add_argument("--save_best", action='store_true',
                        help='save the best model in all of the ckpts')
    args = parser.parse_args()
    args.output_dir = args.output_dir.replace(
        '[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', ''))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    if args.init_model_dir and args.bert_model == '':
        raise Exception('Bert model is needed!')

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    _, _, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

    model = DiverseQA(args)
    args.answer_type = True

    if args.init_model_dir:
        ckpt_file = os.path.join(args.init_model_dir, "bert_base.pt")
        if os.path.exists(ckpt_file):
            ckpt = torch.load(ckpt_file, map_location="cpu")
            state_dict = ckpt["state_dict"]
            model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    logger.warning("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        if args.local_rank in [0]:
            torch.distributed.barrier()
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.warning(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Save the trained model and the tokenizer
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            # Create output directory if needed
            if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(args.output_dir)

            logger.warning("Saving model checkpoint to %s", args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # # Good practice: save your training arguments together with the trained model
            model = torch.nn.DataParallel(model)
            save_model(args, output_dir=args.output_dir, model=model)

            # Load a trained model and vocabulary that you have fine-tuned
            tokenizer = tokenizer_class.from_pretrained(args.output_dir)
            ckpt_file = os.path.join(args.output_dir, "bert_base.pt")
            ckpt = torch.load(ckpt_file, map_location="cpu")
            state_dict = ckpt["state_dict"]
            model.load_state_dict(state_dict)
            model = model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        # Load pretrained model and tokenizer
        save_dir = os.path.join(args.output_dir, ntpath.basename(args.train_file))
        if not args.do_train and not args.not_trained_parallel:
            model = torch.nn.DataParallel(model)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + 'bert_base.pt', recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        if len(checkpoints) > 1:
            final_model = args.output_dir
            tmp_ckpts = []
            for ckpt in checkpoints:
                tail = ckpt.split('-')[-1]
                if tail.isdigit():
                    tmp_ckpts.append(ckpt)
            checkpoints = tmp_ckpts
            print(len(checkpoints))
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
            checkpoints.append(final_model)
        logger.warning("Evaluate the following checkpoints: %s", checkpoints)

        best_f1 = 0.0
        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            ckpt_file = os.path.join(checkpoint, "bert_base.pt")
            ckpt = torch.load(ckpt_file, map_location="cpu")
            state_dict = ckpt["state_dict"]
            model.load_state_dict(state_dict)
            model.to(args.device)

            # Evaluate
            if global_step.isdigit():
                global_step = args.eval_prefix + global_step
            else:
                global_step = args.eval_prefix
            result = evaluate(args, model, tokenizer, prefix=global_step)
            logger.warning("Step {}, Result {}".format(global_step, result))
            for key in result:
                if key.startswith('f1'):
                    if result[key] > best_f1:
                        best_f1 = result[key]
            result = dict((k + ('_{}'.format(global_step) if global_step else ''), v) for k, v in result.items())
            results.update(result)

    logger.warning("Results: {}".format(results))
    pkl_file_path = os.path.join(args.output_dir, '{}.pkl'.format(ntpath.basename(args.train_file)))
    with open(pkl_file_path, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return results


if __name__ == "__main__":
    main()

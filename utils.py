#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 10/12/22 18:16
"""helper functions"""
import os
import random
import time
from datetime import datetime
import re

import torch
import tqdm
import torchtext.vocab as glove_vocab

from torch.nn.utils.rnn import PackedSequence, pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence

### special vocabs
UNK = '<unk>' # `unknown`, for tokens not in the source vocab
BOS = '<bos>' # `beginning of sequence`, for use in CRF
EOS = '<eos>' # `end of sequence`, for use in CRF
PAD = '<pad>' 
device = 'cpu'
non_blocking = False

BATCH_SIZE = 400
### model hyperparams
# embedding dimension
EMBED_DIM     = 50
# LSTM hidden dimension
NUM_HIDDEN    = 50
# number of LSTM layers
NUM_LAYERS    = 2
# whether to make LSTM bidirectional
BIDIRECTIONAL = True

### training hyperparams
LEARNING_RATE = 8e-3
# number of epochs to train
NUM_EPOCHS    = 1
# how many epochs to train before evaluating on dev set (-1 if no eval)
EVAL_EVERY    = 1
# seed for replicability (any arbitrary int)
SEED          = 1334

GLOVE_NORM = 6
GLOVE_DIM = EMBED_DIM

WORD_MAX_LEN = 18 # this entails cnn dim = 8

### data fpaths
TRAINING_TAGGED = '../data/train/ptb_02-21.tagged'
DEV_TAGGED = '../data/dev/ptb_22.tagged'
DEV_SNT = '../data/dev/ptb_22.snt'
TEST_SNT = '../data/test/ptb_23.snt'
# model predictions output filename
PRED_TAGGED = 'preds.tagged'

def display_exec_time(begin: float, msg_prefix: str = ""):
    """displays the script's execution time

    Args:
      begin (float): time stamp for beginning of execution
      msg_prefix (str): display message prefix
    """
    exec_time = time.time() - begin

    msg_header = f'{msg_prefix} finished: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Execution Time:'
    if exec_time > 60:
        et_m, et_s = int(exec_time / 60), int(exec_time % 60)
        print("\n%s %dm %ds" % (msg_header, et_m, et_s))
    else:
        print("\n%s %.2fs" % (msg_header, exec_time))


def get_unique_fpath(fdir, fname):
    idx = 1
    fpath = os.path.join(fdir, fname)
    tail = ext = os.path.splitext(fname)[1]
    while os.path.exists(fpath):
        new_tail = f'_{idx}{ext}'
        fpath = fpath.replace(tail, new_tail)
        tail = new_tail
        idx += 1
    return fpath


def export_preds(snts, preds, output_fname):
    """you are asked to submit your model's predictions on test dataset; the format
    of the export should resemble `.tagged` data, which this function accomplishes

    Args:
      snts:
      preds: model predictions
      output_fname: output filename
    """
    output_fpath = get_unique_fpath('.', output_fname)

    print(f"Exporting predictions at {output_fpath}")
    with open(output_fpath, 'w') as f:
        for snt, pred in zip(snts, preds):
            assert len(snt) == len(pred)
            out_str = " ".join(f'{s}_{p}' for s, p in zip(snt, pred))
            f.write(out_str)
            f.write("\n")


def collect_vocabs(training_data):
    """collects source (sentence) and target (POS tags) vocabs

    `tgt_vocabs_inv` is the inverse of `tgt_vocabs` and is useful for converting
    model predictions (which are originally ints) back to strings

    we collect `src_vocabs` quite naively here; how can we do better? - English, punctuations, numbers
    """
    print("Collecting vocabs")
    src_vocabs_list, tgt_vocabs_list = set(), set()
    for toks, tags in tqdm.tqdm(training_data):
        src_vocabs_list.update(toks)
        tgt_vocabs_list.update(tags)

    src_vocabs_list = sorted(src_vocabs_list)
    tgt_vocabs_list = sorted(tgt_vocabs_list)

    # words (tokens) vocab

    src_vocabs_list = [PAD, UNK, BOS, EOS] + src_vocabs_list
    src_vocabs = {x: i for i, x in enumerate(src_vocabs_list)}  # type: dict

    # POS tags vocab (and its inverse)
    tgt_vocabs, tgt_vocabs_inv = dict(), dict()
    for i, x in enumerate([PAD, UNK, BOS, EOS] + tgt_vocabs_list):
        tgt_vocabs[x] = i
        tgt_vocabs_inv[i] = x

    cache_dir = '.vector_cache/'
    glove = glove_vocab.GloVe(name = '6B', dim = GLOVE_DIM, cache = cache_dir)
    
    # get unk if unknown, so have to do it by myself with random vector

    vocab_to_vec = []

    # doing normalization !!! seems like glove are around 6, so random are set to 6

    for tok in src_vocabs_list:
        if tok in glove.stoi:
            vocab_to_vec.append(glove.get_vecs_by_tokens(tok))
        else:
            r = torch.rand(GLOVE_DIM)
            vocab_to_vec.append(r * GLOVE_NORM / r.norm())
            del r

    return src_vocabs_list, src_vocabs, tgt_vocabs, tgt_vocabs_inv, torch.stack(vocab_to_vec)

def vectorize(data, src_vocabs, tgt_vocabs=None, training = False):
    """converts strings into indices using `src_vocabs` and `tgt_vocabs` if tags are available

    note that tags are first prefixed with <bos>; this is setting up for CRF

    Args:
      data: raw string data  -- data.append((toks_list, tags_list))
      src_vocabs: sentence (source) vocabs
      tgt_vocabs: POS tags (target) vocabs

    Returns:
      tensorized data
    """
    has_tags = tgt_vocabs is not None

    out = []

    if training:
        num_tags = len(tgt_vocabs)
        transition_initial = torch.full((num_tags, num_tags), 1e-4) # circus -11
    
    for toks_tags in data:
        if has_tags:
            toks, tags = toks_tags
            tgt = [tgt_vocabs[x] for x in tags]
        
            if training:
                for i in range(1, len(tgt)):
                    transition_initial[tgt[i], tgt[i-1]] += 1
                transition_initial[tgt[0], tgt_vocabs[BOS]] += 1
                transition_initial[tgt_vocabs[EOS], tgt[-1]] += 1

            # (1, src_len+1)

            # tgt = tgt + [tgt_vocabs[PAD] for _ in range(MAX_LEN - len(tgt))]

            # if len(tgt) > 80:
            #     tgt = tgt[:80]

            tgt_tensor = torch.tensor(tgt, dtype=torch.int64)

        # tensorized source data (sentence tokens)
        if not has_tags:
            toks = toks_tags
        src = [src_vocabs.get(x, src_vocabs[UNK]) for x in toks]
        # (1, src_len)
        
        # because of padding, there must be bos and eos

        # src = src + [src_vocabs[PAD] for _ in range(MAX_LEN - len(src))]

        # if len(src) > 80:
        #     src = src[:80]

        src_tensor = torch.tensor(src, dtype=torch.int64)
        if has_tags:
            out.append((src_tensor, tgt_tensor))
        else:
            out.append((src_tensor, ))
    
    if not training:
        return out
    else:
        transition_initial = torch.log(transition_initial)
        return out, transition_initial


def read_in_gold_data(fpath, max_num_data=-1):
    """gold data's filename ends with `.tagged` and consists of tokenized sentences where each token
    is a tuple of (word, POS tag)

    here we will separate words from POS tags

    Args:
      max_num_data: max number of data to load, useful during debugging

    Returns:
        list of tuples of words and POS tags, each of which in turn is a list
    """
    assert fpath.endswith('tagged')

    with open(fpath) as f:
        data = []
        for i, line in enumerate(f):
            toks_list, tags_list = [], []
            for tok_tag_tuple in line.split():
                toks, tags = tok_tag_tuple.split('_')
                if re.compile(r"[0-9]*(\.)*[0-9]+").fullmatch(toks):
                    toks = "<num>"
                                              # unify number
                toks_list.append(toks.lower())              # use lower case
                tags_list.append(tags)                      
            data.append((toks_list, tags_list))

            if 0 < max_num_data <= i+1:
                break

    return data


def read_in_plain_data(fpath):
    """plain data's filename ends with `.snt` and consists of tokenized sentences

    Returns:
        list of snts, each of which is a list of tokens
    """
    assert fpath.endswith('snt')

    with open(fpath) as f:
        data = [line.split() for line in f.readlines()]

    return data


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

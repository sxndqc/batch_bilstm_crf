#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 10/12/22 17:08
"""script entry

feel free to add your own hyperparameters and change code anywhere

NOTE
    1. regular `.py` files and the jupyter notebook are organized slightly differently,
        although the underlying logic is the same
    2. for PA2, we will assume `batch_size` == 1
    3. default values below are arbitrary and should be modified during experiments
    4. you are asked to submit your model's predictions on test dataset
        * the format of the export should resemble `.tagged` data
        * you will receive test accuracy as part of your feedback
"""

# Should add a dev_loss to see whether under train or over fit


##3#####
"""

Todo:

1. Involve glove. Even after involved it still subject to chage. So unk give them random embed
2. Read LSTM how mask is used

11.27
seems like my nll_loss using log - logexpsum is problematic Solved quickly by log_softmax
mask does not have to be 1 
but the 0 batch pose problems on accuracy, pad is a lot of accuracy

also, there is initial hidden state, so no need for bos tag

and the mps is also problematic, it cannot converge, but cpu can converge
and in bilstm it will nan after half, problem may in the embedding

11.28 mps is problematic because of some misuse of memory
add nonblocking maybe help. if switch between cpu, find them and comment them
https://github.com/pytorch/pytorch/issues/83015

just substitute them with to(device)

of course you can send padding of 0, and mask them
the previous problem is caused by loss function and mps problem
this just add more computation


######

11.28 to do

Should use pack_pad, this save computing resources

and stop using bos for vocab, this is not helpful

and should do the writing problem

"""######
import time
from datetime import datetime

import random
import torch
import torch.optim as optim
import tqdm
import numpy as np

import utils
from model import LSTMCRF, LSTMEncoder
from utils import *


def make_dataset(raw_dataset, test = False):
    
    # print(dataset[0][0].size())

    total_len = len(raw_dataset)
    # print(total_len)

    if not test:
        random.shuffle(raw_dataset)

    dataset = []

    current_index = 0
    current_x_batch = []
    current_y_batch = []

    for data_entry in raw_dataset:
        current_index += 1
        current_x_batch.append(data_entry[0])
        if not test:
            current_y_batch.append(data_entry[1])

        if current_index == BATCH_SIZE:
            if not test:
                dataset.append((current_x_batch, current_y_batch))
            else:
                dataset.append((current_x_batch, ))
            
            del current_x_batch
            current_x_batch = []
            del current_y_batch
            current_y_batch = []
            current_index = 0

    if current_index > 0:
        if not test:
            dataset.append((current_x_batch, current_y_batch))
        else:
            dataset.append((current_x_batch, ))
        del current_x_batch
        current_x_batch = []
        del current_y_batch
        current_y_batch = []
        current_index = 0

    return dataset

def distribution(train, dev, test):
    train_dist = []
    dev_dist = []
    test_dist = []
    for x in train:
        train_dist.append(x[0].shape[1])
    for x in dev:
        dev_dist.append(x[0].shape[1])
    for x in test:
        test_dist.append(x[0].shape[1])
    import matplotlib.pyplot as plt

    plt.figure()
    
    plt.ylabel('nums')
    plt.subplot(311)
    plt.xticks(np.linspace(0,200,201))
    plt.hist(train_dist, range(200))
    plt.subplot(312)
    plt.xticks(np.linspace(0,200,201))
    plt.hist(dev_dist, range(200))
    plt.subplot(313)
    plt.hist(test_dist, range(200))

    plt.show()


def train_simple_lstm(cnn = False):
    name = 'PA3 Training'
    begin = time.time()
    print(f'{name} started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')

    # 1. setup

    utils.set_seed(SEED)

    # 2. data loading
    training_data = utils.read_in_gold_data(TRAINING_TAGGED) #, max_num_data=1000)
    dev_data = utils.read_in_gold_data(DEV_TAGGED) #, max_num_data=100)
    test_data = utils.read_in_plain_data(TEST_SNT)

    # 3. vocabs
    src_vocabs_list, src_vocabs, tgt_vocabs, tgt_vocabs_inv, vocab_to_vec = utils.collect_vocabs(training_data)

    # 4. vectorized data (assume `batch_size` is 1)
    raw_training_dataset, transition_initial = utils.vectorize(training_data, src_vocabs, tgt_vocabs, training= True)
    raw_dev_dataset = utils.vectorize(dev_data, src_vocabs, tgt_vocabs)
    raw_test_dataset = utils.vectorize(test_data, src_vocabs)

    # distribution(raw_training_dataset, raw_dev_dataset, raw_test_dataset)

    train_dataset = make_dataset(raw_training_dataset)
    dev_dataset = make_dataset(raw_dev_dataset)
    test_dataset = make_dataset(raw_test_dataset, test = True)

    # [tensor([[ 895, 1721, 1160,  113, 4494, 3871, 1573, 1447, 1083,    7, 1819,  589,
    #        12,  836, 1448,   18, 1573, 1290,  599, 1432, 5045, 1388, 3254, 1547,
    #       596,   14,    7, 1000,    5,  457,   20,   14, 5045, 4519, 3871,  579,
    #        14, 4080, 2055,  968,  577,   14, 5331, 3713, 1841, 5107,  591,  850,
    #        21]]), tensor([[ 0, 15, 12, 22, 11, 21, 15, 46, 12, 21,  4, 15, 22, 26, 22, 22,  6, 46,
    #      39, 24, 40, 12, 21, 15, 22, 22,  5,  4, 21, 10, 24,  7,  5, 12, 21, 15,
    #      22,  5, 39, 15, 22, 22,  5, 37, 29, 39, 34, 22, 22,  8]])]

    # 5. model init
    tagger = LSTMEncoder(
        num_vocabs=len(src_vocabs),
        src_vocabs_list = src_vocabs_list,
        embed_dim=EMBED_DIM,
        hidden_dim=NUM_HIDDEN,
        num_tags=len(tgt_vocabs),
        num_layers=NUM_LAYERS,
        glove_weights = vocab_to_vec,
        bidirectional=BIDIRECTIONAL,
        cnn = cnn
    )

    tagger.to(device = device, non_blocking = non_blocking)
    
    # 6. optimizer init
    optimizer = optim.Adam(tagger.parameters(), lr=LEARNING_RATE)
    # optimizer = optim.SGD(tagger.parameters(), lr=LEARNING_RATE, momentum = 0.9)

    # 7. training loop
    print("Begin training")
    for epoch in range(1, NUM_EPOCHS+1):
        epoch_loss = 0.
        tagger.train()

        for input_ids, labels in tqdm.tqdm(train_dataset, desc=f'[Epoch {epoch}/{NUM_EPOCHS}]'):


            packed_input_ids = pack_sequence(input_ids, enforce_sorted = False).to(device = device, non_blocking = non_blocking)
            # flattened_labels = torch.cat([labels[i] for i in packed_input_ids.sorted_indices]).unsqueeze(1).to(device = device, non_blocking = non_blocking)
            # if sorting algorithm is different, this could be harmful
            flattened_labels, _, _,_ = pack_sequence(labels, enforce_sorted = False).to(device = device, non_blocking = non_blocking)
            optimizer.zero_grad()                     # clear out gradients from the last step

            loss = tagger.nll_loss(packed_input_ids, flattened_labels.unsqueeze(1)) # forward pass + compute loss
            print(loss)

            loss.backward()                           # backward pass (i.e., computes gradients)
            optimizer.step()                          # update weights

            epoch_loss += loss.item()

            # if tgt_vocabs_inv is not None:
            #     for p, y in zip(input_ids, labels):
            #         print([src_vocabs_list[q] +"_" + tgt_vocabs_inv[x] for q, x in zip(p.tolist(), y.tolist())])
            #         input()

        tagger.eval()
        # display info at the end of epoch
        log = f'[Epoch {epoch}/{NUM_EPOCHS}] Total Loss: {epoch_loss:.2f}'
        if epoch % EVAL_EVERY == 0:
            print("Begin evaluation on dev set")
            _, dev_acc = evaluate_simple_lstm(tagger, dev_dataset, tgt_vocabs_inv, device=device)
            log = f'{log} | Dev Acc: {dev_acc}%'
        print(log)

    # 8. final evaluation on test set
    print("Begin inference on test set")
    test_preds, _ = evaluate_simple_lstm(tagger, test_dataset, tgt_vocabs_inv, device=device)

    # 9. model predictions on test set exported as json
    
    utils.export_preds(test_data, test_preds, PRED_TAGGED)

    utils.display_exec_time(begin, name)

def train(cnn = False):
    name = 'PA3 Training'
    begin = time.time()
    print(f'{name} started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')

    # 1. setup
    utils.set_seed(SEED)

    # 2. data loading
    training_data = utils.read_in_gold_data(TRAINING_TAGGED) #, max_num_data=1000)
    dev_data = utils.read_in_gold_data(DEV_TAGGED) #, max_num_data=100)
    test_data = utils.read_in_plain_data(TEST_SNT)

    # 3. vocabs
    src_vocabs_list, src_vocabs, tgt_vocabs, tgt_vocabs_inv, vocab_to_vec = utils.collect_vocabs(training_data)

    # 4. vectorized data (assume `batch_size` is 1)
    raw_training_dataset, transition_initial = utils.vectorize(training_data, src_vocabs, tgt_vocabs, training= True)
    raw_dev_dataset = utils.vectorize(dev_data, src_vocabs, tgt_vocabs)
    raw_test_dataset = utils.vectorize(test_data, src_vocabs)

    # distribution(raw_training_dataset, raw_dev_dataset, raw_test_dataset)

    train_dataset = make_dataset(raw_training_dataset)
    dev_dataset = make_dataset(raw_dev_dataset)
    test_dataset = make_dataset(raw_test_dataset, test = True)

    # 5. model init
    tagger = LSTMCRF(
        src_vocabs=src_vocabs,
        src_vocabs_list = src_vocabs_list,
        tgt_vocabs=tgt_vocabs,
        embed_dim=EMBED_DIM,
        hidden_dim=NUM_HIDDEN,
        num_layers=NUM_LAYERS,
        glove_weights = vocab_to_vec,
        transition_initial = transition_initial,
        bidirectional=BIDIRECTIONAL,
        cnn = cnn
    )

    tagger.to(device = device, non_blocking = non_blocking)
    
    # 6. optimizer init
    optimizer = optim.Adam(tagger.parameters(), lr=LEARNING_RATE)
    # optimizer = optim.SGD(tagger.parameters(), lr=LEARNING_RATE, momentum = 0.9)

    # test_preds, _ = evaluate(tagger, test_dataset, tgt_vocabs_inv, device=device)
    # input()
    # 7. training loop
    print("Begin training")
    for epoch in range(1, NUM_EPOCHS+1):
        epoch_loss = 0.
        tagger.train()

        for input_ids, labels in tqdm.tqdm(train_dataset, desc=f'[Epoch {epoch}/{NUM_EPOCHS}]'):

            packed_input_ids = pack_sequence(input_ids, enforce_sorted = False).to(device = device, non_blocking = non_blocking)
            flattened_labels, _, _,_ = pack_sequence(labels, enforce_sorted = False).to(device = device, non_blocking = non_blocking)
            optimizer.zero_grad()                     # clear out gradients from the last step

            loss = tagger.nll_loss(packed_input_ids, labels) # forward pass + compute loss
            print(loss)

            loss.backward()                           # backward pass (i.e., computes gradients)
            optimizer.step()                          # update weights

            epoch_loss += loss.item()

            # if tgt_vocabs_inv is not None:
            #     for p, y in zip(input_ids, labels):
            #         print([src_vocabs_list[q] +"_" + tgt_vocabs_inv[x] for q, x in zip(p.tolist(), y.tolist())])
            #         input()

        tagger.eval()
        # display info at the end of epoch
        log = f'[Epoch {epoch}/{NUM_EPOCHS}] Total Loss: {epoch_loss:.2f}'
        if epoch % EVAL_EVERY == 0:
            print("Begin evaluation on dev set")
            _, dev_acc = evaluate(tagger, dev_dataset, tgt_vocabs_inv, device=device)
            log = f'{log} | Dev Acc: {dev_acc}%'
        print(log)

    # 8. final evaluation on test set
    print("Begin inference on test set")
    test_preds, _ = evaluate(tagger, test_dataset, tgt_vocabs_inv, device=device)

    # 9. model predictions on test set exported as json
    
    utils.export_preds(test_data, test_preds, PRED_TAGGED)

    utils.display_exec_time(begin, name)

def evaluate_simple_lstm(tagger, eval_dataset, tgt_vocabs_inv=None, device=None):

    dev_loss = num_correct = num_tags = acc = 0
    labels = None

    preds_list = []
    with torch.no_grad():
        for eval_data in tqdm.tqdm(eval_dataset):
            # dev has labels but test doesn't

            # eval_data = ()
            input_ids = eval_data[0]
            labels = eval_data[1] if len(eval_data) == 2 else None

            packed_input_ids = pack_sequence(input_ids, enforce_sorted = False).to(device = device, non_blocking = non_blocking)
            preds, preds_len = tagger.decode(packed_input_ids)

            if labels is not None:
                # flattened_labels = torch.cat([labels[i] for i in packed_input_ids.sorted_indices]).unsqueeze(1).to(device = device, non_blocking = non_blocking)
                flattened_labels, _, _, _ = pack_sequence(labels, enforce_sorted = False).to(device = device, non_blocking = non_blocking)
                print("Dev Loss: ", tagger.nll_loss(packed_input_ids, flattened_labels.unsqueeze(1)))

            if labels is not None:
                for pred, l, lb in zip(preds, preds_len, labels):
                    num_tags += len(lb)
                    num_correct += torch.sum(pred[:l] == lb)

            if torch.is_tensor(preds):
                assert preds.ndim == 2
                preds = [preds[i, :l].tolist() for i, l in enumerate(preds_len)]

            if tgt_vocabs_inv is not None and isinstance(preds[0][0], int):
                for y in preds:
                    preds_list.append([tgt_vocabs_inv[x] for x in y])

            """The mismatch problem is raised from unmatch for labels and source, as the source is shuffled.
            And zip will not raise error"""

    # if test, `acc` will be 0
    if labels is not None:
        acc = round(int(num_correct) * 100 / int(num_tags), 2)

    return preds_list, acc

def evaluate(tagger, eval_dataset, tgt_vocabs_inv=None, device=None):

    dev_loss = num_correct = num_tags = acc = 0
    labels = None

    preds_list = []
    with torch.no_grad():
        for eval_data in tqdm.tqdm(eval_dataset):
            # dev has labels but test doesn't

            # eval_data = ()
            input_ids = eval_data[0]
            labels = eval_data[1] if len(eval_data) == 2 else None

            packed_input_ids = pack_sequence(input_ids, enforce_sorted = False).to(device = device, non_blocking = non_blocking)
            preds= tagger.decode(packed_input_ids)

            # this preds is the same shape as labels

            if labels is not None:
                print("Dev Loss: ", tagger.nll_loss(packed_input_ids, labels))

            if labels is not None:
                for pred, lb in zip(preds, labels):
                    num_tags += len(lb)
                    num_correct += torch.sum(pred == lb)

            preds = [p.tolist() for p in preds]

            if tgt_vocabs_inv is not None and isinstance(preds[0][0], int):
                for y in preds:
                    preds_list.append([tgt_vocabs_inv[x] for x in y])

    # if test, `acc` will be 0
    if labels is not None:
        acc = round(int(num_correct) * 100 / int(num_tags), 2)

    return preds_list, acc


if __name__ == '__main__':
    
    # to use vallina bilstm, change the name;

    train(cnn = True)

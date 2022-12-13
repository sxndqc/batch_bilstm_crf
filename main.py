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

    total_len = len(raw_dataset)
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
    name = 'Training BiLSTM'
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
    name = 'Training CRF'
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

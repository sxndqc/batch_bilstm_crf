# Batch_First == True cause LSTM wrong output, should be false to avoid this.
# https://github.com/pytorch/pytorch/issues/80306

import torch
import torch.nn as nn

from utils import *

def mask(input_ids):
    return (input_ids > 0).float()

class LSTMEncoder(nn.Module):
    def __init__(self, num_vocabs, src_vocabs_list, embed_dim, hidden_dim, num_tags, num_layers, glove_weights,
                 bidirectional=False, cnn = False):
        super().__init__()
        self.src_vocabs_list = src_vocabs_list
        self.using_cnn = cnn

        if self.using_cnn:
            self.conv1 = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size=2)
            self.pooling1 = torch.nn.MaxPool1d(kernel_size = 3, stride = 2)   
            self.cnn_dim = ((WORD_MAX_LEN - 2 + 1) - 3) // 2 + 1
        else:
            self.cnn_dim = 0

        # self.embedding = nn.Embedding(num_vocabs, embed_dim)
        self.embedding = torch.nn.Embedding.from_pretrained(glove_weights, freeze= False, padding_idx = 0)
        self.lstm = nn.LSTM(embed_dim + self.cnn_dim, hidden_dim, num_layers, batch_first=True,
                            bidirectional=bidirectional)
        self.linear = nn.Linear(hidden_dim * (bidirectional + 1), num_tags)
        
            
    def forward(self, packed_input_ids):

        original_input_embeds, batch_sizes, sorted_indices, unsorted_indices = packed_input_ids
        
        if self.using_cnn:
            words = [self.src_vocabs_list[i] for i in original_input_embeds]
            ascii_words = [[float(ord(c)) for c in word[:WORD_MAX_LEN]] + [0 for _ in range(WORD_MAX_LEN - len(word))] for word in words]
        
            ascii_words = torch.tensor(ascii_words, requires_grad = False).view(len(words), 1, WORD_MAX_LEN)
            cnn_embeds = torch.relu(self.pooling1(self.conv1(ascii_words))).view(len(words), self.cnn_dim)

            input_embeds = torch.cat((self.embedding(original_input_embeds), cnn_embeds), dim=1)
        else:
            input_embeds = self.embedding(original_input_embeds)
        
        lstm_input = PackedSequence(input_embeds, batch_sizes, sorted_indices, unsorted_indices)
        lstm_hidden, _ = self.lstm(lstm_input)
        del lstm_input
        
        lstm_output, batch_sizes_2, sorted_indices_2, unsorted_indices_2 = lstm_hidden
        assert (batch_sizes == batch_sizes_2).all()
        assert (sorted_indices == sorted_indices_2).all()
        assert (unsorted_indices == unsorted_indices_2).all()
        
        lstm_output = self.linear(lstm_output)

        emission = PackedSequence(lstm_output, batch_sizes, sorted_indices, unsorted_indices)

        del lstm_output
        
        return emission


    def decode(self, packed_input_ids):

        emission, batch_sizes, sorted_indices, unsorted_indices = self(packed_input_ids)
        packed_outputs = torch.log_softmax(emission, dim = 1).argmax(dim = 1)

        assert packed_outputs.ndim == 1

        outputs, outputs_lengths = pad_packed_sequence(PackedSequence(packed_outputs, batch_sizes, sorted_indices, unsorted_indices))
        return outputs.mT, outputs_lengths


    def nll_loss(self, packed_input_ids, flattened_labels):

        emission, batch_sizes, sorted_indices, unsorted_indices = self(packed_input_ids)
        outputs = - torch.log_softmax(emission, dim = 1)

        num_tokens = outputs.size(0)

        assert num_tokens == len(flattened_labels)

        # The next step is to extract according to labels (can we reorganize labels with sorted? and sum)
        return torch.sum(torch.gather(input = outputs, dim = 1, index = flattened_labels), dim = 0) / num_tokens


class LSTMCRF(nn.Module):

    def __init__(self, src_vocabs, src_vocabs_list, tgt_vocabs, embed_dim, hidden_dim, num_layers, glove_weights,
                 transition_initial, bidirectional=False, cnn = False):
        """see `LSTMEncoder` above for description of arguments"""
        super().__init__()

        self.src_vocabs = src_vocabs
        self.tgt_vocabs = tgt_vocabs
        self.num_tags = len(tgt_vocabs)

        self.lstm = LSTMEncoder(
            num_vocabs=len(src_vocabs),
            src_vocabs_list = src_vocabs_list,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_tags=len(tgt_vocabs),
            num_layers=num_layers,
            glove_weights = glove_weights,
            bidirectional=bidirectional,
            cnn = cnn
        )
        #self.transitions = nn.Parameter(torch.rand(self.num_tags, self.num_tags))  # this can be pre-trained
        self.transitions = nn.Parameter(transition_initial)

        self.transitions.data[self.tgt_vocabs[BOS], :] = -1000.
        self.transitions.data[:, self.tgt_vocabs[EOS]] = -1000.


    def forward(self, emission, emission_length):
        
        this_batch = emission.size(0)
        max_length = emission.size(1)
        assert emission.size(2) == self.num_tags

        alphas = []

        alpha = self.transitions[:, self.tgt_vocabs[BOS]].expand(this_batch, self.num_tags) + emission[:, 0, :]

        # alpha and feat broadcast in different direction; at least valid label length is 1
        alphas.append(alpha) # alpha at position 0

        for i in range(1, max_length):
            feat = emission[:, i, :]  # batch * tag
            this_emission_to_add = feat.expand(self.num_tags, this_batch, self.num_tags).permute(1,2,0)
            add_3 = this_emission_to_add + self.transitions + alpha.expand(self.num_tags, this_batch, self.num_tags).permute(1,0,2)

            alpha = torch.logsumexp(add_3, dim = 2)

            alphas.append(alpha)

        real_alphas = [] # batch * 1

        for i, l in enumerate(emission_length):

            real_alphas.append(torch.logsumexp(alphas[int(l)-1][i] + self.transitions[self.tgt_vocabs[EOS], :], dim = 0)) # sum up at the real terminal position

        return sum(real_alphas)


    def decode(self, input_ids):

        packed_emission = self.lstm(input_ids)

        emission, emission_length = pad_packed_sequence(packed_emission)
        emission = emission.permute(1,0,2)

        this_batch = emission.size(0)
        max_length = emission.size(1)
        assert emission.size(2) == self.num_tags

        s = torch.full((this_batch, self.num_tags), -10000.)
        s[:, self.tgt_vocabs[BOS]] = 0.

        score_history = []
        preds = []

        for i in range(emission.size(1)):
            new_s = s.expand(self.num_tags, this_batch, self.num_tags).permute(1,0,2) + \
                self.transitions.expand(this_batch, -1, -1) + \
                emission[:, i, :].expand(self.num_tags, this_batch, self.num_tags).permute(1,2,0)
            # from s, to emission

            s, last_index = new_s.max(dim = 2)

            score_history.append(s)
            preds.append(last_index) # a batch is added

        score_history = torch.stack(score_history).permute(1,0,2)
        preds = torch.stack(preds).permute(1,0,2)  # batch * max_length * tag

        real_preds = []

        for i, l in enumerate(emission_length):
            
            final_score = score_history[i, l-1, :] + self.transitions[self.tgt_vocabs[EOS], :]
            pos_max = final_score.argmax(dim = 0)  # the largest in the final round, which is the largest path
            pos = l-1

            real_pred = []
            real_pred.append(int(pos_max))

            while pos > 0:
                real_pred.append(int(preds[i, pos, real_pred[-1]]))
                pos -= 1
            
            assert len(real_pred) == l

            real_pred.reverse()
            real_preds.append(torch.LongTensor(real_pred))

        return real_preds


    def score(self, emission_sequence, labels_sequence):
        
        s = 0

        for labels in labels_sequence:
            s += self.transitions[labels[0], self.tgt_vocabs[BOS]]
            for i in range(1, len(labels)):
                s += self.transitions[labels[i], labels[i-1]]
            s += self.transitions[self.tgt_vocabs[EOS], labels[-1]]
        
        flattened_labels, _, _,_ = pack_sequence(labels_sequence, enforce_sorted = False).to(device = device, non_blocking = non_blocking)

        s += torch.sum(torch.gather(input = emission_sequence, dim = 1, index = flattened_labels.unsqueeze(1)), dim = 0).squeeze()
        return s
        

    def nll_loss(self, input_ids, labels):

        packed_emission = self.lstm(input_ids)

        emission, emission_length = pad_packed_sequence(packed_emission)
        
        return (self.forward(emission.permute(1,0,2), emission_length) - self.score(packed_emission.data, labels)) / len(labels)

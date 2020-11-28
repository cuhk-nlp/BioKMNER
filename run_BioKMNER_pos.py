from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling_NER_kv_concate import (CONFIG_NAME, WEIGHTS_NAME,
                                                            BertConfig,
                                                            BertForTokenClassification)
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import datetime
from data_helper import stanford_feature_processor
now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
log_file_name = './logs/log-'+now_time
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    filename = log_file_name,
                    filemode = 'w',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

# output_dir = './results/result-' + now_time

logger = logging.getLogger(__name__)

# number_of_labels = 103

def eval_sentence(y_pred, y, sentence, word2id):
    words = sentence.split(' ')
    seg_true = []
    seg_pred = []
    word_true = ''
    word_pred = ''

    y_word = []
    y_pos = []
    y_pred_word = []
    y_pred_pos = []
    for y_label, y_pred_label in zip(y, y_pred):
        y_word.append(y_label[0])
        y_pos.append(y_label[2:])
        y_pred_word.append(y_pred_label[0])
        y_pred_pos.append(y_pred_label[2:])

    for i in range(len(y_word)):
        word_true += words[i]
        word_pred += words[i]
        if y_word[i] in ['S', 'E']:
            pos_tag_true = y_pos[i]
            word_pos_true = word_true + '_' + pos_tag_true
            if word_true not in word2id:
                word_pos_true = '*' + word_pos_true + '*'
            seg_true.append(word_pos_true)
            word_true = ''
        if y_pred_word[i] in ['S', 'E']:
            pos_tag_pred = y_pred_pos[i]
            word_pos_pred = word_pred + '_' + pos_tag_pred
            seg_pred.append(word_pos_pred)
            word_pred = ''

    seg_true_str = ' '.join(seg_true)
    seg_pred_str = ' '.join(seg_pred)
    return seg_true_str, seg_pred_str


def input2file(save_path):
    s = input("Please input demo sentence: \n")
    f = open(save_path, "w")
    for item in s:
        I = 'O'
        f.write(item + '' + I + '\n')
    f.write('\n ')


def pos_evaluate_word_PRF(y_pred, y):
    #dict = {'E': 2, 'S': 3, 'B':0, 'I':1}
    y_word = []
    y_pos = []
    y_pred_word = []
    y_pred_pos = []
    for y_label, y_pred_label in zip(y, y_pred):
        y_word.append(y_label[0])
        y_pos.append(y_label[2:])
        y_pred_word.append(y_pred_label[0])
        y_pred_pos.append(y_pred_label[2:])

    word_cor_num = 0
    pos_cor_num = 0
    yp_wordnum = y_pred_word.count('E')+y_pred_word.count('S')
    yt_wordnum = y_word.count('E')+y_word.count('S')
    start = 0
    for i in range(len(y_word)):
        if y_word[i] == 'E' or y_word[i] == 'S':
            word_flag = True
            pos_flag = True
            for j in range(start, i+1):
                if y_word[j] != y_pred_word[j]:
                    word_flag = False
                    pos_flag = False
                    break
                if y_pos[j] != y_pred_pos[j]:
                    pos_flag = False
            if word_flag:
                word_cor_num += 1
            if pos_flag:
                pos_cor_num += 1
            start = i+1

    wP = word_cor_num / float(yp_wordnum) if yp_wordnum > 0 else -1
    wR = word_cor_num / float(yt_wordnum) if yt_wordnum > 0 else -1
    wF = 2 * wP * wR / (wP + wR)
    logger.info('word P: %f' % wP)
    logger.info('word R: %f' % wR)
    logger.info('word F: %f' % wF)

    # pP = pos_cor_num / float(yp_wordnum) if yp_wordnum > 0 else -1
    # pR = pos_cor_num / float(yt_wordnum) if yt_wordnum > 0 else -1
    # pF = 2 * pP * pR / (pP + pR)

    pP = precision_score(y, y_pred)
    pR = recall_score(y, y_pred)
    pF = f1_score(y, y_pred)

    logger.info('POS P: %f' % pP)
    logger.info('POS R: %f' % pR)
    logger.info('POS F: %f' % pF)
    return (wP, wR, wF), (pP, pR, pF)


def pos_evaluate_OOV(y_pred_list, y_list, sentence_list, word2id):
    word_cor_num = 0
    pos_cor_num = 0
    yt_wordnum = 0

    y_word_list = []
    y_pos_list = []
    y_pred_word_list = []
    y_pred_pos_list = []
    for y_label, y_pred_label in zip(y_list, y_pred_list):
        y_word = []
        y_pos = []
        y_pred_word = []
        y_pred_pos = []
        for y_l in y_label:
            y_word.append(y_l[0])
            y_pos.append(y_l[2:])
        for y_pred_l in y_pred_label:
            y_pred_word.append(y_pred_l[0])
            y_pred_pos.append(y_pred_l[2:])
        y_word_list.append(y_word)
        y_pos_list.append(y_pos)
        y_pred_word_list.append(y_pred_word)
        y_pred_pos_list.append(y_pred_pos)

    for y_w, y_p, y_p_w, y_p_p, sentence in zip(y_word_list, y_pos_list, y_pred_word_list, y_pred_pos_list, sentence_list):
        start = 0
        for i in range(len(y_w)):
            if y_w[i] == 'E' or y_w[i] == 'S':
                word = ''.join(sentence[start:i+1])
                if word in word2id:
                    start = i + 1
                    continue
                word_flag = True
                pos_flag = True
                yt_wordnum += 1
                for j in range(start, i+1):
                    if y_w[j] != y_p_w[j]:
                        word_flag = False
                        pos_flag = False
                        break
                    if y_p[j] != y_p_p[j]:
                        pos_flag = False
                if word_flag:
                    word_cor_num += 1
                if pos_flag:
                    pos_cor_num += 1
                start = i + 1

    word_OOV = word_cor_num / float(yt_wordnum) if yt_wordnum > 0 else -1
    pos_OOV = pos_cor_num / float(yt_wordnum) if yt_wordnum > 0 else -1
    logger.info('# of OOV: %d' % yt_wordnum)
    logger.info('# of correct word: %d' % word_cor_num)
    logger.info('# of correct pos: %d' % pos_cor_num)
    logger.info('word OOV: %f' % word_OOV)
    logger.info('POS OOV: %f' % pos_OOV)
    return word_OOV, pos_OOV


def get_word2id(data_dir):
    word2id_path = os.path.join(data_dir, 'word2id.json')
    if os.path.exists(word2id_path):
        print('load word from existing file')
        with open(word2id_path, 'r', encoding='utf8') as f:
            return json.loads(f.readline())
    word2id = {'<PAD>': 0}
    word = ''
    index = 1
    for line in open(os.path.join(data_dir, "train.tsv")):
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            continue
        line=line.strip()
        splits = line.split('\t')
        character = splits[0]
        label = splits[-1][0]
        word += character
        if label in ['S', 'E']:
            if word not in word2id:
                word2id[word] = index
                index += 1
            word = ''
    with open(word2id_path, 'w', encoding='utf8') as f:
        json.dump(word2id, f, ensure_ascii=False)
        f.write('\n')
    return word2id


def get_wordpos2id(data_dir, lower_threshold, upper_threshold):
    word2count_path = os.path.join(data_dir, 'wordpos2count.json')
    wordpos2count = {}
    if os.path.exists(word2count_path):
        print('load word pos from existing file')
        with open(word2count_path, 'r', encoding='utf8') as f:
            wordpos2count = json.loads(f.readline())
    else:
        word = ''
        for line in open(os.path.join(data_dir, "train.tsv")):
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                continue
            splits = line.split('\t')
            character = splits[0]
            labels = splits[-1][:-1].split('-')
            cws_label = labels[0]
            pos_label = labels[1]
            word += character
            if cws_label in ['S', 'E']:
                word = word + '_' + pos_label
                if word not in wordpos2count:
                    wordpos2count[word] = 1
                else:
                    wordpos2count[word] += 1
                word = ''
        with open(word2count_path, 'w', encoding='utf8') as f:
            json.dump(wordpos2count, f, ensure_ascii=False)
            f.write('\n')
    wordpos2id = {'<PAD>': 0}
    word2pos = {}
    index = 1
    for word, count in wordpos2count.items():
        if lower_threshold < count < upper_threshold:
            wordpos2id[word] = index
            index += 1
            wp = word.split('_')
            w = wp[0]
            p = wp[1]
            if w in word2pos:
                if p not in word2pos[w]:
                    word2pos[w].append(p)
            else:
                word2pos[w] = [p]
    return wordpos2id, word2pos


def get_testword2id(data_dir):
    word2id = {'<PAD>': 0}
    word = ''
    index = 1
    for line in open(os.path.join(data_dir, "test.tsv")):
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            continue
        splits = line.split('\t')
        character = splits[0]
        label = splits[-1][0]
        word += character
        if label in ['S', 'E']:
            if word not in word2id:
                word2id[word] = index
                index += 1
            word = ''
    return word2id


class NER(BertForTokenClassification):

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None, attention_mask_label=None,
                word_seq=None, feature_seq=None, word_matrix=None, feature_matrix=None, number_of_labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        batch_size,max_len,feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32,device='cuda')
        # for i in range(batch_size):
        #     jj = -1
        #     for j in range(max_len):
        #         if valid_ids[i][j].item() == 1:
        #             jj += 1
        #             valid_output[i][jj] = sequence_output[i][j]
        for i in range(batch_size):
            temp=sequence_output[i][valid_ids[i] == 1]
            valid_output[i][:temp.size(0)]=temp


        # word_memory_output = self.word_memory(word_seq, sequence_output, label_value_matrix, word_mask)
        #

        # word_attention = self.word_attention(word_seq, valid_output, word_matrix)
        # feature_attention = self.feature_attention(feature_seq, valid_output, feature_matrix)
        word_memory_output = self.word_memory(word_seq, valid_output, feature_seq, feature_matrix)

        # conc = torch.cat([valid_output, word_attention, feature_attention], dim=2)

        conc = self.dropout(word_memory_output)
        # sequence_output, _ = self.bilstm_encoder(conc)

        logits = self.classifier(conc)

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        loss_fct = CrossEntropyLoss(ignore_index=0)
        total_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++====
        # crf = CRF(tagset_size=number_of_labels+1, gpu=True)
        # total_loss = crf.neg_log_likelihood_loss(logits, attention_mask, labels)
        # scores, tag_seq = crf._viterbi_decode(logits, attention_mask)
        # Only keep active parts of the loss
        return total_loss, logits


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, word=None,
                 syn_feature=None, word_matrix=None, syn_matrix=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.word = word
        self.word_matrix = word_matrix
        self.syn_matrix = syn_matrix
        self.syn_feature = syn_feature


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None,
                 word_ids=None, syn_feature_ids=None, word_matching_matrix=None, syn_matching_matrix=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.word_ids = word_ids
        self.syn_feature_ids = syn_feature_ids
        self.word_matching_matrix = word_matching_matrix
        self.syn_matching_matrix = syn_matching_matrix


def readfile(filename, flag):
    f = open(filename)
    data = []
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
            continue
        splits = line.split('\t')
        char = splits[0]
        l = splits[-1][:-1]

        sentence.append(char)
        label.append(l)

    if len(sentence) >0:
        data.append((sentence,label))
        sentence = []
        label = []
    return data


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None, flag='test'):
        """Reads a tab separated value file."""
        return readfile(input_file, flag)

class PosProcessor(DataProcessor):
    """Processor for the cws POS CTB5 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv"), flag='train'), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv"), flag='dev'), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv"), flag='test'), "test")

    def get_labels(self):
        #return ["NT", "JJ", "NR", "PU", "NN", "[CLS]", "[SEP]"]
        #return ['CD', 'SB', 'DER', 'IJ', 'NR', 'CS', 'MSP', 'NN', 'LC', 'VV', 'M', 'OD', 'VE', 'AD', 'DT', 'PU', 'ETC', 'NT', 'SP','NP', 'PN', 'P', 'VP', 'VC', 'VA', 'DEC', 'FW', 'AS', 'X', 'DEG', 'BA', 'DEV', 'CC', 'JJ', 'LB', "[CLS]", "[SEP]"]
        return ['O', 'B-NR', 'E-NR', 'B-NN', 'E-NN', 'S-CC', 'B-VV', 'E-VV', 'I-NN', 'B-NT', 'E-NT', 'S-NN', 'S-PU',
                'I-NR', 'S-LC', 'S-AS', 'S-ETC', 'S-DEC', 'B-CD', 'I-CD', 'E-CD', 'S-M', 'S-DEG', 'B-JJ', 'E-JJ',
                'S-VC', 'S-CD', 'I-JJ', 'B-AD', 'E-AD', 'S-AD', 'S-JJ', 'S-P', 'S-PN', 'B-VA', 'E-VA', 'S-DEV',
                'S-VV', 'B-LC', 'E-LC', 'B-DT', 'E-DT', 'S-SB', 'B-OD', 'E-OD', 'B-P', 'E-P', 'S-VE', 'S-DT', 'B-M',
                'E-M', 'B-CS', 'E-CS', 'B-PN', 'E-PN', 'S-VA', 'I-NT', 'I-AD', 'I-M', 'B-CC', 'E-CC', 'S-OD', 'S-MSP',
                'S-NR', 'S-BA', 'I-VV', 'B-FW', 'I-FW', 'E-FW', 'B-PU', 'E-PU', 'S-CS', 'S-NT', 'I-OD', 'S-LB', 'I-VA',
                'B-ETC', 'E-ETC', 'B-VE', 'E-VE', 'I-P', 'B-NP', 'E-NP', 'S-DER', 'S-SP', 'B-SP', 'E-SP', 'I-PU',
                'I-PN', 'I-CC', 'B-IJ', 'E-IJ', 'I-DT', 'B-MSP', 'E-MSP', 'S-IJ', 'S-X', 'B-VC', 'I-VC', 'E-VC',
                'S-FW', 'I-CS', 'S-NP', 'S-VP', "[CLS]", "[SEP]"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class NER_attention_Processor(DataProcessor):

    def __init__(self, data_dir, feature_flag):
        self.data_dir = data_dir
        self.label_list = self._get_labels()
        self.sfp = stanford_feature_processor(self.data_dir)
        self.feature_flag = feature_flag
        self.number_of_labels = len(self.label_list) - 3
        self.gram2id, self.feature2id = self.get_feature2id()

    def get_train_examples(self):
        """See base class."""
        data_path = os.path.join(self.data_dir, "train.tsv")
        lines = self._read_tsv(data_path, flag='train')
        all_feature_data = self.sfp.read_features(flag='train')
        data = self._get_data(lines, all_feature_data)
        return self._create_examples(data, "train")

    def get_dev_examples(self):
        """See base class."""
        data_path = os.path.join(self.data_dir, "dev.tsv")
        lines = self._read_tsv(data_path, flag='dev')
        all_feature_data = self.sfp.read_features(flag='dev')
        data = self._get_data(lines, all_feature_data)
        return self._create_examples(data, "dev")

    def get_test_examples(self):
        """See base class."""
        data_path = os.path.join(self.data_dir, "test.tsv")
        lines = self._read_tsv(data_path, flag='test')
        all_feature_data = self.sfp.read_features(flag='test')
        data = self._get_data(lines, all_feature_data)
        return self._create_examples(data, "test")

    def get_labels(self):
        #return ["NT", "JJ", "NR", "PU", "NN", "[CLS]", "[SEP]"]
        #return ['CD', 'SB', 'DER', 'IJ', 'NR', 'CS', 'MSP', 'NN', 'LC', 'VV', 'M', 'OD', 'VE', 'AD', 'DT', 'PU', 'ETC', 'NT', 'SP','NP', 'PN', 'P', 'VP', 'VC', 'VA', 'DEC', 'FW', 'AS', 'X', 'DEG', 'BA', 'DEV', 'CC', 'JJ', 'LB', "[CLS]", "[SEP]"]
        return self.label_list

    def _get_labels(self):
        label_path = os.path.join(self.data_dir, 'label2id')
        label_list = ['OO']
        with open(label_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            label_list.append(line.split('\t')[0])
        label_list.extend(['[CLS]', '[SEP]'])
        return label_list

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label, word_list, syn_feature_list) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            word = '#*#*#'.join(word_list)
            label = label
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, word=word,
                             syn_feature=syn_feature_list))
        return examples

    def get_feature2id(self, min_threshold=1):
        all_feature2count = self.sfp.read_feature2count()
        gram2count = all_feature2count['gram2count']
        if self.feature_flag == 'pos':
            feature2count = all_feature2count['pos_tag2count']
        elif self.feature_flag == 'chunk':
            feature2count = all_feature2count['chunk_tag2count']
        elif self.feature_flag == 'dep':
            feature2count = all_feature2count['dep_tag2count']
        else:
            raise ValueError()
        gram2id = {'<PAD>': 0, '<UNK>': 1}
        feature2id = {'<PAD>': 0, '<UNK>': 1}
        gram_index = 2
        feature_index = 2
        for gram, count in gram2count.items():
            if count > min_threshold:
                gram2id[gram] = gram_index
                gram_index += 1
        for feature, count in feature2count.items():
            if count > min_threshold:
                feature2id[feature] = feature_index
                feature_index += 1
        return gram2id, feature2id

    def _get_data(self, lines, all_feature_data):
        data = []
        if self.feature_flag == 'pos':
            for (sentence, label), feature_list in zip(lines, all_feature_data):
                word_list = []
                syn_feature_list = []
                # word_matching_position = []
                # syn_matching_position = []
                # print(len(self.gram2id))
                # print(len(self.feature2id))
                for token_index, token in enumerate(feature_list):
                    current_token_pos = token['pos']
                    current_token = token['word']
                    current_feature = current_token + '_' + current_token_pos
                    if current_token not in self.gram2id:
                        current_token = '<UNK>'
                    if current_feature not in self.feature2id:
                        if current_token_pos not in self.feature2id:
                            current_feature = '<UNK>'
                        else:
                            current_feature = current_token_pos
                    word_list.append(current_token)
                    syn_feature_list.append(current_feature)

                    assert current_token in self.gram2id
                    assert current_feature in self.feature2id


                data.append((sentence, label, word_list, syn_feature_list))


        return data

countnumber_sentence=0
countnumber_input=0
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, gram2id, feature2id, max_word_size):
    global countnumber_sentence, countnumber_input
    """Loads a data file into a list of `InputBatch`s."""
    print(label_list)
    label_map = {label: i for i, label in enumerate(label_list, 1)}
    print(label_map)

    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        wordlist = example.word
        wordlist = wordlist.split('#*#*#') if len(wordlist) > 0 else []
        syn_features = example.syn_feature
        tokens = []
        labels = []
        valid = []
        label_mask = []

        word_ids = []
        feature_ids = []
        word_matching_matrix = np.zeros((max_seq_length, max_word_size), dtype=np.int)
        syn_matching_matrix = np.zeros((max_seq_length, max_word_size), dtype=np.int)

        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)

        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]


        if len(wordlist) > max_word_size:
            wordlist = wordlist[:max_word_size]
            syn_features = syn_features[:max_word_size]

        if len(wordlist)==len(syn_features):
            for word in wordlist:
                # print(wordlist)
                if word == '':
                    continue
                try:
                    # if word not in gram2id:
                    #     word_ids.append(gram2id["<UNK>"])
                    # else:
                    word_ids.append(gram2id[word])
                except KeyError:
                    print(word)
                    print(wordlist)
                    print(textlist)
                    raise KeyError()
            for feature in syn_features:
                # print(syn_features)
                # print(feature)
                feature_ids.append(feature2id[feature])
            sentence_len=len(word_ids)
            while len(word_ids) < max_word_size:
                word_ids.append(0)
                feature_ids.append(0)
        else:
            countnumber_sentence+=1
            while len(word_ids) < max_word_size:
                word_ids.append(0)
                feature_ids.append(0)


        for i in range(sentence_len):
                begin_char_index = max(i - 1, 0)
                end_char_index = min(i + 2, sentence_len)
                for w in range(begin_char_index, end_char_index):
                    word_matching_matrix[i + 1][w] = 1
                    syn_matching_matrix[i + 1][w] = 1

        ntokens = []
        segment_ids = []
        label_ids = []

        ntokens.append("[CLS]")
        segment_ids.append(0)

        valid.insert(0, 1)
        label_mask.insert(0, 1)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")

        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])

        input_ids = tokenizer.convert_tokens_to_ids(ntokens)

        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        if len(input_ids) < max_seq_length:
            countnumber_input +=1
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        # print(len(feature_ids))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length
        assert len(word_ids) == max_word_size
        assert len(feature_ids) == max_word_size

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info(
        #         "syntax_ids: %s" % " ".join([str(x) for x in feature_ids]))
        #     logger.info(
        #         "martix_0: %s" % " ".join([str(x) for x in word_matching_matrix[0]]))
        #     logger.info(
        #         "martix_1: %s" % " ".join([str(x) for x in word_matching_matrix[1]]))
        #     logger.info(
        #         "martix_2: %s" % " ".join([str(x) for x in word_matching_matrix[2]]))
        #     logger.info(
        #         "martix_3: %s" % " ".join([str(x) for x in word_matching_matrix[3]]))

            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          valid_ids=valid,
                          label_mask=label_mask,
                          word_ids=word_ids,
                          syn_feature_ids=feature_ids,
                          word_matching_matrix=word_matching_matrix,
                          syn_matching_matrix=syn_matching_matrix))
    return features


def main():

    def _str2bool(s):
        if s.lower() in ['true', 'yes']:
            return True
        elif s.lower() in ['false', 'no']:
            return False
        else:
            raise argparse.ArgumentTypeError('Wrong bool type!')


    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")

    # parser.add_argument("--output_dir",
    #                    default=output_dir,
    #                    type=str,
    #                    help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=150,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_word_size",
                        default=75,
                        type=int,
                        help="The maximum candidate word size used by attention. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--restore',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--patient', type=int, default=3, help="Can be used for distant debugging.")
    parser.add_argument('--word_num_threshold', type=int, default=0, help="Can be used for distant debugging.")
    parser.add_argument('--word_num_upper_threshold', type=int, default=100000000, help="Can be used for distant debugging.")
    parser.add_argument('--feature_flag', type=str, default='pos', help="")
    args = parser.parse_args()
    logger.info(args)

    output_dir = './results/result-' + now_time + "-" + args.bert_model
    args.output_dir = output_dir

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    
    # processors = {"pos": PosProcessor, 'posattention': Pos_attention_Processor}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    


    word2id = get_word2id(args.data_dir)
    logger.info('# of word: %d' % len(word2id))
    # logger.info('# of used word-pos in memory: %d' % len(wordpos2id))
    # logger.info('# of used word in attention: %d' % len(gram2id))
    processor = NER_attention_Processor(args.data_dir, args.feature_flag)
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1
    logger.info('# of labels: %d' % (num_labels-4))

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples()
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    
    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))
    model = NER.from_pretrained(args.bert_model,
                                cache_dir=cache_dir,
                                num_labels=num_labels,
                                args=args)
    if args.restore:
        logger.info('Restoring model from %s' % args.bert_model)
        model.load_state_dict(torch.load(os.path.join(args.bert_model, 'pytorch_model.bin')))
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

#AdamW
    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias',  'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    #     ]
    #
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    # warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
    #                                          t_total=num_train_optimization_steps)
    # if args.fp16:
    #     try:
    #         from apex import amp
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    #

#FUSED_BERT_ADAM
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight','word_memory.word_embedding_c.weight','word_memory.word_embedding_a.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                             t_total=num_train_optimization_steps)

    else:
        #num_train_optimization_steps=-1
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    label_map = {i : label for i, label in enumerate(label_list,1)}

    best_epoch = -1
    best_wp = -1
    best_wr = -1
    best_wf = -1
    best_woov = -1
    best_pp = -1
    best_pr = -1
    best_pf = -1
    best_poov = -1
    history = {'epoch': [], 'word_p': [], 'word_r': [], 'word_f': [], 'word_oov': [],
               'pos_p': [], 'pos_r': [], 'pos_f': [], 'pos_oov': []}
    num_of_no_improvement = 0
    patient = args.patient

    if args.do_train:
        train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer,
                                                      processor.gram2id, processor.feature2id, args.max_word_size)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
        all_word_ids = torch.tensor([f.word_ids for f in train_features], dtype=torch.long)
        all_feature_ids = torch.tensor([f.syn_feature_ids for f in train_features], dtype=torch.long)
        all_word_matching_matrix = torch.tensor([f.word_matching_matrix for f in train_features], dtype=torch.float)
        all_syn_matching_matrix = torch.tensor([f.syn_matching_matrix for f in train_features], dtype=torch.float)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids, all_lmask_ids,
                                   all_word_ids, all_feature_ids, all_word_matching_matrix, all_syn_matching_matrix)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask, word_ids, \
                feature_ids, word_matching_matrix, feature_matching_matrix = batch
                loss, _ = model(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask, word_ids,
                                feature_ids, word_matching_matrix, feature_matching_matrix, num_labels-4)

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            model.to(device)

            if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                eval_examples = processor.get_test_examples()
                eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer,
                                                             processor.gram2id, processor.feature2id, args.max_word_size)

                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)
                all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
                all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
                all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
                all_word_ids = torch.tensor([f.word_ids for f in eval_features], dtype=torch.long)
                all_feature_ids = torch.tensor([f.syn_feature_ids for f in eval_features], dtype=torch.long)
                all_word_matching_matrix = torch.tensor([f.word_matching_matrix for f in eval_features],
                                                        dtype=torch.float)
                all_syn_matching_matrix = torch.tensor([f.syn_matching_matrix for f in eval_features],
                                                       dtype=torch.float)
                eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_valid_ids, all_lmask_ids,
                                          all_word_ids, all_feature_ids, all_word_matching_matrix, all_syn_matching_matrix)
                # Run prediction for full data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
                model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0
                y_true = []
                y_pred = []
                label_map = {i: label for i, label in enumerate(label_list, 1)}
                for input_ids, input_mask, segment_ids, label_ids,valid_ids,l_mask, word_ids, feature_ids, \
                    word_matching_matrix, feature_matching_matrix in tqdm(eval_dataloader, desc="Evaluating"):

                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    valid_ids = valid_ids.to(device)
                    label_ids = label_ids.to(device)
                    l_mask = l_mask.to(device)
                    word_ids = word_ids.to(device)
                    feature_ids = feature_ids.to(device)
                    word_matching_matrix = word_matching_matrix.to(device)
                    feature_matching_matrix = feature_matching_matrix.to(device)

                    with torch.no_grad():
                        _, logits = model(input_ids, segment_ids, input_mask, labels=label_ids, valid_ids=valid_ids,
                                          attention_mask_label=l_mask, word_seq=word_ids, feature_seq=feature_ids,
                                          word_matrix=word_matching_matrix, feature_matrix=feature_matching_matrix,
                                          number_of_labels=num_labels-4)

                    logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    input_mask = input_mask.to('cpu').numpy()

                    for i, label in enumerate(label_ids):
                        temp_1 = []
                        temp_2 = []
                        for j, m in enumerate(label):
                            if j == 0:
                                continue
                            elif label_ids[i][j] == num_labels - 1:
                                y_true.append(temp_1)
                                y_pred.append(temp_2)
                                break
                            else:
                                temp_1.append(label_map[label_ids[i][j]])
                                if logits[i][j] == 0:
                                    pred_label='OO'
                                else:
                                    pred_label = label_map [logits[i][j]]
                                temp_2.append(pred_label)

                output_eval_file = os.path.join(args.output_dir, "eval_results.txt")

                # the evaluation method of cws
                y_true_all = []
                y_pred_all = []
                sentence_all = []
                for y_true_item in y_true:
                    y_true_all += y_true_item
                for y_pred_item in y_pred:
                    y_pred_all += y_pred_item
                for example, y_true_item in zip(eval_examples, y_true):
                    sen = example.text_a
                    sen = sen.strip()
                    sen = sen.split(' ')
                    if len(y_true_item) != len(sen):
                        print(len(sen))
                        sen = sen[:len(y_true_item)]
                    sentence_all.append(sen)
                (wp, wr, wf), (pp, pr, pf) = pos_evaluate_word_PRF(y_pred_all, y_true_all)
                woov, poov = pos_evaluate_OOV(y_pred, y_true, sentence_all, word2id)
                history['epoch'].append(epoch)
                history['word_p'].append(wp)
                history['word_r'].append(wr)
                history['word_f'].append(wf)
                history['word_oov'].append(woov)
                history['pos_p'].append(pp)
                history['pos_r'].append(pr)
                history['pos_f'].append(pf)
                history['pos_oov'].append(poov)
                logger.info("=======entity level========")
                # logger.info("Epoch: %d, word P: %f, word R: %f, word F: %f, word OOV: %f",
                #             epoch + 1, wp, wr, wf, woov)
                # logger.info("Epoch: %d,  P: %f,  R: %f,  F: %f",
                #             epoch + 1, pp, pr, pf)
                # logger.info("=======entity level========")
                # the evaluation method of NER
                report = classification_report(y_true, y_pred, digits=4)
                with open(output_eval_file, "a") as writer:
                    logger.info("***** Eval results *****")
                    logger.info("=======token level========")
                    logger.info("\n%s", report)
                    logger.info("=======token level========")
                    writer.write(report)

                if pf > best_pf:
                    best_epoch = epoch + 1
                    best_wp = wp
                    best_wr = wr
                    best_wf = wf
                    best_woov = woov
                    best_pp = pp
                    best_pr = pr
                    best_pf = pf
                    best_poov = poov
                    num_of_no_improvement = 0

                    # with open(os.path.join(args.output_dir, 'POS_result.txt'), "w") as writer:
                    #     writer.write("Epoch: %d, word P: %f, word R: %f, word F: %f, word OOV: %f" %
                    #                  (epoch + 1, wp, wr, wf, woov))
                    #     writer.write("Epoch: %d,  pos P: %f,  pos R: %f,  pos F: %f,  pos OOV: %f" %
                    #                  (epoch + 1, pp, pr, pf, poov))
                    #     for i in range(len(y_pred)):
                    #         sentence = eval_examples[i].text_a
                    #         seg_true_str, seg_pred_str = eval_sentence(y_pred[i], y_true[i], sentence, word2id)
                    #         # logger.info("true: %s", seg_true_str)
                    #         # logger.info("pred: %s", seg_pred_str)
                    #         writer.write('True: %s\n' % seg_true_str)
                    #         writer.write('Pred: %s\n\n' % seg_pred_str)

                    # Save a trained model and the associated configuration
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                    with open(output_config_file, 'w') as f:
                        f.write(model_to_save.config.to_json_string())
                    label_map = {i: label for i, label in enumerate(label_list, 1)}
                    model_config = {"bert_model": args.bert_model, "do_lower": args.do_lower_case,
                                    "max_seq_length": args.max_seq_length, "num_labels": len(label_list) + 1,
                                    "label_map": label_map}
                    json.dump(model_config, open(os.path.join(args.output_dir, "model_config.json"), "w"))
                    # Load a trained model and config that you have fine-tuned
                else:
                    num_of_no_improvement += 1

            if num_of_no_improvement >= patient:
                logger.info('\nEarly stop triggered at epoch %d\n' % (epoch + 1))
                break

        # logger.info("\n=======best f entity level========")
        # logger.info("Epoch: %d, word P: %f, word R: %f, word F: %f, word OOV: %f",
        #             best_epoch, best_wp, best_wr, best_wf, best_woov)
        # logger.info("Epoch: %d,  pos P: %f,  pos R: %f,  pos F: %f,  pos OOV: %f",
        #             best_epoch, best_pp, best_pr, best_pf, best_poov)
        # logger.info("\n=======best f entity level========")

        with open(os.path.join(args.output_dir, 'history.json'), 'w', encoding='utf8') as f:
            json.dump(history, f)
            f.write('\n')
        

if __name__ == "__main__":
    main()

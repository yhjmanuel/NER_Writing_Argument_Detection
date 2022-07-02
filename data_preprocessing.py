from typing import NamedTuple, Sequence, Any, List
import string
from utils import *
import pandas as pd
import spacy
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from transformers import LongformerForTokenClassification, AutoTokenizer
import numpy as np
import math
import torch
import torch.nn as nn
import random
import pickle


class Config:
    """
    Congfiguration of the data processor. Indicates the directory of 
    data, as well as the type of tokenizer that will be used.
    """
    data_dir = 'feedback-prize-2021/train/'
    csv_dir = 'feedback-prize-2021/train.csv'
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    # if we are using roberta model instead of longformer, 
    # use this line of code to indicate our tokenizer
    # tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')


class StudentWritingDatasetProcessor:
    """
    a class that can process & return different forms of annotated data (for 
    training in pycrfsuite, pytorch, etc.) and can visualize the named entities of 
    a given article. The visualization function is only available if you use Jupyter Notebook.
    """

    def __init__(self, data_dir, csv_dir, tokenizer):
        # set the data directory
        self.data_dir = data_dir
        self.csv_dir = csv_dir
        # read the csv data, which contains information about the mentions of every
        # article in the training set
        self.csv = pd.read_csv(self.csv_dir)
        # get a set of all article's ids
        self.doc_ids = set(self.csv['id'])
        # with self.data, we can find the text of an article given its id (as key)
        self.data = {}
        self.label_to_num = {OUTSIDE: 0}
        self.get_label_dict()
        self.num_to_label = {item: key for key, item in enumerate(self.label_to_num)}
        # read all the text files here
        self.read_all_txt()
        # if we use pycrfsuite, we can directly use BIO-encoded label. 
        # We can call processor.annotated_data to get this form of data
        # Otherwise, we might need to numerize the labels
        self.annotated_data = []
        # set our tokenizer
        self.tokenizer = tokenizer

    # create a dict that maps a label to an integer
    def get_label_dict(self):
        num = 1
        for C in CLASSES:
            self.label_to_num[BEGIN + DELIM + C] = num
            num += 1
            self.label_to_num[INSIDE + DELIM + C] = num
            num += 1
    
    # read a single article, helper method
    def read_single_txt(self, doc_id):
        temp_txt = ''
        with open(self.data_dir + doc_id + '.txt') as file:
            for line in file:
                temp_txt += line
        self.data[doc_id] = temp_txt

    # read all the articles
    def read_all_txt(self):
        for doc_id in self.doc_ids:
            self.read_single_txt(doc_id)

    
    # save each document as an AnnotatedDoc object, which records the tokens
    # and mentions of a document. Save all objects in a list, which can be accessed
    # using object.annotated_data
    def create_annotated_data(self):
       
        print("---------------------Converting labels to BIO-based, this may take a minute---------------------")
        for doc_id in self.doc_ids:
            self.annotate_single_txt(doc_id)
        print("---------------------------Labels successfully converted to BIO-based---------------------------")

    # helper method, convert a document to an AnnotatedDoc object
    # in this method, we tokenize a document, and assign a 'BIO' label to each token
    # then we convert the 'BIO' labels to Mention objects
    # finally, with the tokens and the Mention objects, we can create an AnnotatedDoc object for an article
    def annotate_single_txt(self, doc_id):
        temp_start = 0
        temp_end = 0
        doc = self.data[doc_id]
        tokens, mentions = [], []
        temp_csv = self.csv[self.csv['id'] == doc_id].reset_index(drop=True)
        # make sure that there is at least one label in the doc
        if len(temp_csv) > 0:
            for i in range(len(temp_csv)):
                # check if there is a gap (namely 'O') between two mentions / a gap before the first mention
                if temp_end != int(temp_csv['discourse_end'][i]) - 1:
                    # there is a gap
                    gap = doc[temp_end: int(temp_csv['discourse_start'][i])].split()
                    tokens.extend(gap)
                    mentions.extend([OUTSIDE] * len(gap))
                # update temp_start
                temp_start = int(temp_csv['discourse_start'][i])
                temp_end = int(temp_csv['discourse_end'][i])
                temp_tokens = doc[temp_start: temp_end].split()
                tokens.extend(temp_tokens)
                if temp_csv['discourse_end'][i] == 'Concluding Statement':
                    temp_mention_type = 'CS'
                else:
                    temp_mention_type = temp_csv['discourse_type'][i]
                mentions.extend([BEGIN + DELIM + temp_mention_type])
                mentions.extend([INSIDE + DELIM + temp_mention_type] * (len(temp_tokens) - 1))
            # after iterate over all mentions, we check whether there are tokens after the last mention
            if temp_end < len(doc):
                last_gap = doc[temp_end: len(doc)].split()
                tokens.extend(last_gap)
                mentions.extend([OUTSIDE] * len(last_gap))
            self.annotated_data.append(AnnotatedDoc(doc_id, tokens, decode_bio(mentions)))

    # visualize a doc and its entities using methods in scipy
    # default set jupyter = True
    # this method only works if you are using jupyter notebook
    def visualize_doc(self, doc_id, jupyter=True):
        assert doc_id in self.data
        doc = self.data[doc_id]
        # reset index, or our index will not match the ones in the dataframe
        temp_csv = self.csv[self.csv['id'] == doc_id].reset_index(drop=True)
        mention_range = []
        labels = list(set(temp_csv.discourse_type))
        for i in range(len(temp_csv)):
            # save the entities to a list
            mention_range.append({'start': int(temp_csv['discourse_start'][i]),
                                  'end': int(temp_csv['discourse_end'][i]),
                                  'label': temp_csv['discourse_type'][i]})
        # choose the color for each entity
        colors = {'Rebuttal': 'limegreen', 'Position': 'yellow', 'Claim': 'aquamarine', 'Lead': 'tomato',
                  'Evidence': 'plum', 'Counterclaim': 'pink', 'Concluding Statement': 'orange'}
        # plot the text and the entities
        spacy.displacy.render({'text': doc, 'ents': mention_range},
                              style='ent',
                              options={'ents': labels, 'colors': colors},
                              manual=True, jupyter=jupyter)

    # convert the labels to numbers given the label dict
    def numerize_labels(self, labels):
        numerized = []
        for label in labels:
            label = label.upper()
            if label == 'B-CONCLUDING STATEMENT':
                numerized.append(self.label_to_num['B-CS'])
            elif label == 'I-CONCLUDING STATEMENT':
                numerized.append(self.label_to_num['I-CS'])
            else:
                numerized.append(self.label_to_num[label])
        return numerized

    # convert the annotated data list, which consists of AnnotatedDoc objects,
    # to a list of dicts which can be sent to Roberta / Longformer pretrained token classification model
    # each dict contains "input_ids", "attention_mask", and "labels"
    def make_tokenizer_input(self):
        data_strings = [" ".join(self.annotated_data[i].tokens) for i in range(len(self.annotated_data))]
        encoding = self.tokenizer(data_strings, padding='max_length', truncation=True, return_offsets_mapping=True)
        labels = []
        for i in range(len(data_strings)):
            temp_labels = self.numerize_labels(
                encode_bio(self.annotated_data[i].tokens, self.annotated_data[i].mentions))
            labels.append(temp_labels)
        # create a list of dictionaries. Within each dictionary, we can find
        # the index of every char in a sentence and the label that corresponds to the char
        char_label_dic_list = []
        # total complexity: number of chars
        for k in range(len(self.annotated_data)):
            char_label_dic = {}
            curr_idx = 0
            sen = self.annotated_data[k].tokens
            for i in range(len(sen)):
                if i == 0:
                    for j in range(curr_idx, curr_idx + len(sen[i])):
                        char_label_dic[j] = labels[k][i]
                    curr_idx += len(sen[i])
                else:
                    for j in range(curr_idx, curr_idx + len(sen[i]) + 1):
                        char_label_dic[j] = labels[k][i]
                    curr_idx = curr_idx + len(sen[i]) + 1
            char_label_dic_list.append(char_label_dic)

        # initialize a list (of dics):
        tokenizer_input = []
        # get 'max length' of each sequence (here is will be the default 'max_length' 
        # of the tokenizer that we use
        seq_len = len(encoding['offset_mapping'][0])
        # we have len(data_strings) artivles in total, create labels for each article's sub-token
        for i in range(len(data_strings)):
            # create seq_len labels for each sequence
            # pytorch's CrossEntropyLoss's default setting: ignore_index = -100
            encoded_labels = [-100] * seq_len
            # assign the label for each sub-token here
            for j in range(seq_len):
                # if encoding['offset_mapping'][i][j][1] == 0, the subtoken in this position is a special token
                if encoding['offset_mapping'][i][j][1] != 0:
                    encoded_labels[j] = char_label_dic_list[i][
                        int((encoding['offset_mapping'][i][j][0] + encoding['offset_mapping'][i][j][1]) / 2)]
            temp_data = {}
            temp_data['input_ids'] = torch.tensor(encoding['input_ids'][i])
            temp_data['attention_mask'] = torch.tensor(encoding['attention_mask'][i])
            temp_data['labels'] = torch.tensor(encoded_labels)
            tokenizer_input.append(temp_data)

        return tokenizer_input

if __name__ == "__main__":
    processor = StudentWritingDatasetProcessor(data_dir=Config.data_dir,
                                               csv_dir=Config.csv_dir,
                                               tokenizer=Config.tokenizer)
    processor.create_annotated_data()
    print("---------------------Starting to generate a pickle file (Our model's input)---------------------")
    model_input = processor.make_tokenizer_input()
    # the generated data will be save using pickle
    file_name = "longformer_model_input_tensor.pkl"
    open_file = open(file_name, "wb")
    # this will be a file of approximately 1.55GB (for longformer tokenizer)
    # or 200 MB (for roberta tokenizer) 
    # generated in your project directory
    # you can use this pickle file later for easier training
    pickle.dump(model_input, open_file)
    print("-----------------------The pickle file is successfully generated and saved----------------------")
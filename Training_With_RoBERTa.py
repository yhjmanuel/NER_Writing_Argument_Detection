import pandas as pd
import pickle
from collections import Counter
from typing import NamedTuple, Sequence, Any, List
import string
from utils import *
import pandas as pd
import spacy
from transformers import RobertaTokenizerFast
from transformers import RobertaForTokenClassification
import numpy as np
import math
import torch
import torch.nn as nn
import random
from sklearn.metrics import f1_score

class Config:
    """
    Set the training configurations.
    """
    n_classes = 15
    n_epochs = 3
    lr = 1e-5
    model = RobertaForTokenClassification.from_pretrained('roberta-base', num_labels=15)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              factor=0.9,
                                                              mode="min",
                                                              patience=10,
                                                              cooldown=10,
                                                              min_lr=5e-6,
                                                              verbose=True)
    train_batch_size = 16
    dev_batch_size = 16
    test_batch_size = 16
    train_split = 0.8
    data_dir = 'project_data_and_models/feedback-prize-2021/train/'
    csv_dir = 'project_data_and_models/feedback-prize-2021/train.csv'
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    device = 'cuda' if torch.cuda.is_available else 'cpu'

if __name__ == '__main__':
    # train, dev, test data obtained by processing data_processing.py's output pickle file
    with open('project_data_and_models/roberta_train_set.pickle', 'rb') as train:
        train_data = pickle.load(train)
    with open('project_data_and_models/roberta_dev_set.pickle', 'rb') as dev:
        dev_data = pickle.load(dev)
    with open('project_data_and_models/roberta_test_set.pickle', 'rb') as test:
        test_data = pickle.load(test)
    train_set = DataLoader(train_data, batch_size=Config.train_batch_size, shuffle=True, pin_memory=True)
    dev_set = DataLoader(dev_data, batch_size=Config.dev_batch_size, shuffle=True, pin_memory=True)
    test_set = DataLoader(test_data, batch_size=Config.test_batch_size, shuffle=True, pin_memory=True)
    trainer = Trainer(Config, train_set, dev_set, test_set)
    trainer.train(save_model_path='roberta_model.pt')
    # token level metric
    trainer.run_on_dev_or_test(dataset='test')
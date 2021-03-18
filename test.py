import json
import random
import os
import logging
import pickle
import string
from pathlib import Path
from collections import Counter, OrderedDict, defaultdict as ddict
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import util
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForMaskedLM
import argparse
from args import get_train_test_args

args = get_train_test_args()
model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

context_path = f'{args.train_dir}/{args.train_datasets}_context_encodings.pt'
data = util.load_pickle(context_path)
# check encodings
for i in range(10):
    print(tokenizer.decode(data['input_ids'][i]))

masked_data = util.mask_train_data(data, tokenizer)
for i in range(10):
    print(tokenizer.decode(masked_data['input_ids'][i]))


# data_dir = "datasets/indomain_val"
# datasets = 'squad,nat_questions,newsqa'
# dataset_name = '_squad_nat_questions_newsqa'
# util.save_context("datasets/indomain_train", ['sample'], '_sample')
# with open(f'{data_dir}/{dataset_name}') as f:
#     sentences = f.readlines()
# corpus = [s.strip() for s in sentences]
# print(corpus)


# util.encode_context_data(tokenizer, 'datasets/indomain_train', '_sample')
# cache_path = 'datasets/indomain_train/_sample_context_encodings.pt'
# tokenized_examples = util.load_pickle(cache_path)
# for ids in tokenized_examples["input_ids"]:
#     print(tokenizer.decode(ids))
# masked_examples = util.mask_train_data(tokenized_examples, tokenizer, mask_prob=0.15)
# mask_cache_path = 'datasets/indomain_val/_newsqa_masked_encodings.pt'
# util.save_pickle(masked_examples, mask_cache_path)
# masked_examples = util.load_pickle(mask_cache_path)
# print(masked_examples)
# print(masked_examples['input_ids'].size())

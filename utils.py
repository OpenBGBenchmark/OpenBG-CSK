# coding: UTF-8
from torch.utils.data import TensorDataset, DataLoader
import csv
import torch
from tqdm import tqdm
import time, os
from datetime import timedelta
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader

PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]'
sepa, sepb, sepc = '[unused1]', '[unused2]', '[unused3]'


def get_segment_ids(tokenized_sent):
    segments_ids = []
    segment = 0
    for word in tokenized_sent:
        segments_ids.append(segment)
        if word == '[SEP]':
            segment += 1
    return segments_ids


def load_dataset(path, config):
    contents = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            if (len(lin.split('\t'))) >= 3:
                t1, t2, label = lin.split('\t')[:3]
                t1 = t1.replace(' ', '')
                t2 = t2.replace(' ', '')
                r = "适用"
                raw_sent = '[SEP]'.join([t1, r, t2])
                contents.append([raw_sent, t1, t2, int(label)])
    return contents


def build_dataset(config):
    train = load_dataset(config.train_path, config)
    dev = load_dataset(config.dev_path, config)
    test = load_dataset(config.test_path, config)
    return train, dev, test


def build_iterator(dataset, config, istrain):
    sent = torch.LongTensor([temp[0] for temp in dataset])
    labels = torch.FloatTensor([temp[1] for temp in dataset])
    train_dataset = TensorDataset(sent, labels)
    if istrain:
        train_loader = DataLoader(dataset, shuffle=True, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, shuffle=False, batch_size=config.batch_size, num_workers=config.num_workers, drop_last=True)
    return train_loader


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def gettoken(config, sent):
    tokenizer = config.tokenizer
    encode_result = tokenizer(sent, padding='max_length', truncation=True, max_length=config.max_length)
    input_ids = torch.tensor(encode_result['input_ids'])
    attention_mask = torch.tensor(encode_result['attention_mask'])
    type_ids = torch.tensor(encode_result['token_type_ids'])
    position_ids = []
    for j, ids in enumerate(input_ids):
        position_id = list(range(config.max_length))
        position_ids.append(position_id)
    position_ids = torch.tensor(position_ids)
    return input_ids, attention_mask, type_ids, position_ids


if __name__ == "__main__":
   print("")

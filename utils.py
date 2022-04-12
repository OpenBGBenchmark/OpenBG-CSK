# coding: UTF-8
from torch.utils.data import TensorDataset, DataLoader
import csv
import torch
from tqdm import tqdm
import time, os
from datetime import timedelta
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
import json

PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]'


def load_dataset(path, config):
    contents = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            line_dict = json.loads(lin)
            subject = line_dict["subject"]
            object = line_dict["object"]
            predicate = line_dict["predicate"]
            triple_id = line_dict["triple_id"]
            raw_sent = SEP.join([subject, predicate, object])
            if "salience" in line_dict.keys():
                salience = line_dict["salience"]
                contents.append([raw_sent, triple_id, int(salience)])
            else:
                contents.append([raw_sent, triple_id, 0])
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
        train_loader = DataLoader(dataset, shuffle=True, batch_size=config.batch_size, num_workers=config.num_workers,
                                  drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, shuffle=False, batch_size=config.batch_size,
                                  num_workers=config.num_workers, drop_last=True)
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

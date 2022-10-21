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

PAD, CLS, SEP, T = '[PAD]', '[CLS]', '[SEP]', '<T>'


def load_local_dataset(path, config):
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
            if "salience" in line_dict.keys():
                salience = line_dict["salience"]
                contents.append([triple_id, subject, object, predicate, int(salience)])
            else:
                contents.append([triple_id, subject, object, predicate, 0])
    return contents


def build_dataset(config):
    train = load_local_dataset(config.train_path, config)
    dev = load_local_dataset(config.dev_path, config)
    test = load_local_dataset(config.test_path, config)
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


def gettoken(config, subjects, objects, predicates):
    tokenizer = config.tokenizer
    sents = []
    for s, o, p in zip(subjects, objects, predicates):
        raw_sent = SEP.join([s, p, o])
        sents.append(raw_sent)
    encode_result = tokenizer(sents, padding='max_length', truncation=True, max_length=config.max_length)
    input_ids = torch.tensor(encode_result['input_ids'])
    attention_mask = torch.tensor(encode_result['attention_mask'])
    type_ids = torch.tensor(encode_result['token_type_ids'])
    position_ids = []
    for j, ids in enumerate(input_ids):
        position_id = list(range(config.max_length))
        position_ids.append(position_id)
    position_ids = torch.tensor(position_ids)
    return input_ids, attention_mask, type_ids, position_ids


def gettoken_pmi(config, subjects, objects, predicates):
    tokenizer = config.tokenizer
    sents = []
    subs = []
    objs = []
    b = config.b_new_tokens
    a = config.a_new_tokens
    for s, o, p in zip(subjects, objects, predicates):
        if "适用" in p:
            r = p.split("_")[0] + T * b + "适用" + T * b + p.split("_")[2]
        elif "搭配" in p:
            r = p.split("_")[0] + T * b + "搭配" + T * b + p.split("_")[2]
        else:
            r = p.split("_")[0] + T * b + "经常" + T * b + p.split("_")[2]
        raw_sent = T * a + s + T * a + r + T * a + o + T * a + '.'
        sents.append(raw_sent)
        subs.append(s)
        objs.append(o)

    tokenizer = config.tokenizer
    encode_result = tokenizer(sents, padding='max_length', truncation=True, max_length=config.max_length)
    input_ids = torch.tensor(encode_result['input_ids'])
    attention_mask = torch.tensor(encode_result['attention_mask'])
    # print(masked_head)
    masked_head = []
    masked_tail = []
    masked_both = []
    for j in range(len(sents)):
        sent_p = f"{CLS} {sents[j]}{SEP}"
        tokenized_sent = tokenizer.tokenize(sent_p)
        tokenized_t1 = tokenizer.tokenize(subs[j])
        tokenized_t2 = tokenizer.tokenize(objs[j])

        # mask sentence
        masked_sent_list = mask_sentence(tokenized_sent, tokenized_t1, tokenized_t2)

        indexed_masked_list = [tokenizer.convert_tokens_to_ids(m) for m in masked_sent_list]
        # head tail both
        while len(indexed_masked_list[0]) < config.max_length:
            indexed_masked_list[0].append(0)
            indexed_masked_list[1].append(0)
            indexed_masked_list[2].append(0)
        masked_head.append(indexed_masked_list[0])
        masked_tail.append(indexed_masked_list[1])
        masked_both.append(indexed_masked_list[2])

    masked_head = torch.tensor(masked_head)
    masked_tail = torch.tensor(masked_tail)
    masked_both = torch.tensor(masked_both)
    return input_ids, attention_mask, masked_head, masked_tail, masked_both


def mask_sentence(tokenized_sent, tokenized_t1, tokenized_t2):
    masked_sent_list = []
    # mask head
    masked_sent_list.append(mask(tokenized_sent, tokenized_t2))
    # mask tail
    masked_sent_list.append(mask(tokenized_sent, tokenized_t1))
    # mask both
    tokenized_sent = mask(tokenized_sent, tokenized_t1)
    masked_sent_list.append(mask(tokenized_sent, tokenized_t2))
    # print(masked_sent_list,tokenized_t1,tokenized_t2)
    return masked_sent_list


def mask(tokenized_sent, tokenized_to_mask):
    tokenized_masked = tokenized_sent.copy()
    for idx_sent in range(len(tokenized_masked) - len(tokenized_to_mask)):
        match = []
        for idx_mask in range(len(tokenized_to_mask)):
            match.append(tokenized_masked[idx_sent + idx_mask] == tokenized_to_mask[idx_mask])
        if all(match):
            for idx_mask in range(len(tokenized_to_mask)):
                if tokenized_masked[idx_sent + idx_mask] not in ['the', 'a', 'an']:
                    tokenized_masked[idx_sent + idx_mask] = '[MASK]'
            break
    return tokenized_masked


if __name__ == "__main__":
    print("")

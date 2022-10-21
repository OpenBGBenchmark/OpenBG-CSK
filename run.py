# coding: UTF-8
import time, os
import numpy as np
from train_eval import train, test
import random
from model import PMI, KGBERT, Config
import argparse
from utils import build_dataset, build_iterator, get_time_dif, gettoken, load_local_dataset, gettoken_pmi
import datasets
from torch.utils.data import DataLoader
import torch
import json

parser = argparse.ArgumentParser(description='Salient triple classification')
parser.add_argument("--do_train", type=bool, default=False, help="Whether to run training.",)
parser.add_argument("--test_batch", default=200, type=int, help="Test every X updates steps.")

parser.add_argument("--data_dir", default="data", type=str, help="The task data directory.")
parser.add_argument("--model_dir", default="bert-base-chinese", type=str, help="The directory of pretrained models")
parser.add_argument("--output_dir", default='output/save_dict/', type=str, help="The path of result data and models to be saved.")
parser.add_argument('--model', type=str, default="PMI",  help="model type")
parser.add_argument('--a_new_tokens', type=int, default=2, help='a_new_tokens')
parser.add_argument('--b_new_tokens', type=int, default=2, help='b_new_tokens')
parser.add_argument("--lamda", type=float, default=0.5, help="weight between neccisity and sufficiency",)

# models param
parser.add_argument("--max_length", default=256, type=int, help="the max length of sentence.")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size for training.")
parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
parser.add_argument("--dropout", default=0.1, type=float, help="Drop out rate")
parser.add_argument("--epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument('--seed', type=int, default=1, help="random seed for initialization")
parser.add_argument('--hidden_size', type=int, default=768,  help="random seed for initialization")

args = parser.parse_args()


def train_entry():
    start_time = time.time()
    print("Loading data...")
    # train_data_all = load_dataset(config.train_path, config)
    # train_data_all = dataset["train[:10%]"]
    # offset = int(len(train_data_all) * 0.1)
    dev_data = datasets.load_dataset("Yincen/SalienceEvaluation", split='train[:10%]')
    train_data = datasets.load_dataset("Yincen/SalienceEvaluation", split='train[10%:]')
    print(len(train_data))
    print(len(dev_data))
    test_data = load_local_dataset(config.test_path, config)
    train_iter = DataLoader(
        train_data,
        shuffle=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True)
    dev_iter = DataLoader(dev_data, shuffle=False, batch_size=config.batch_size,
                          num_workers=config.num_workers, drop_last=False)
    test_iter = DataLoader(test_data, shuffle=False, batch_size=config.batch_size,
                           num_workers=config.num_workers, drop_last=False)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    # train
    if args.model == "PMI":
        model = PMI(config).to(config.device)
    else:
        model = KGBERT(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)


def test_entry():
    test_data = load_local_dataset(config.test_path, config)
    if args.model == "PMI":
        model = PMI(config).to(config.device)
    else:
        model = KGBERT(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path+"model.ckpt"))
    model.eval()
    loader = DataLoader(test_data, shuffle=False, batch_size=config.batch_size)
    predicts = []
    for i, batches in enumerate(loader):
        triple_id, subject, object, predicate, label = batches
        labels = [float(x) for x in label]
        labels = np.asarray(labels)
        labels = torch.from_numpy(labels)
        if config.model == "PMI":
            input_ids, attention_mask, masked_head, masked_tail, masked_both = gettoken_pmi(config, subject, object, predicate)
            input_ids, masked_head, masked_tail, masked_both, labels, attention_mask = input_ids.to(
                config.device), masked_head.to(config.device), masked_tail.to(
                config.device), masked_both.to(config.device), labels.to(config.device), attention_mask.to(
                config.device)
            pmi = model(input_ids, masked_head, masked_tail, masked_both, attention_mask)
        else:
            input_ids, attention_mask, type_ids, position_ids = gettoken(config, subject, object, predicate)
            input_ids, attention_mask, type_ids, labels = \
                input_ids.to(config.device), attention_mask.to(config.device), type_ids.to(
                    config.device), labels.to(config.device)
            position_ids = position_ids.to(config.device)
            pmi = model(input_ids, attention_mask, type_ids, position_ids)
        bires = torch.where(pmi > 0.2, torch.tensor([1]).cuda(), torch.tensor([0]).cuda())
        for b, t in zip(bires, triple_id):
            predicts.append({"salience": b.item(), "triple_id": t})

    with open(config.save_path + "OpenBG-CSK_test.jsonl", "w") as f:
        for t in predicts:
            f.write(json.dumps(t, ensure_ascii=False)+"\n")


if __name__ == '__main__':
    config = Config(args)
    torch.cuda.empty_cache()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    if not args.do_train:
        test_entry()
    else:
        train_entry()


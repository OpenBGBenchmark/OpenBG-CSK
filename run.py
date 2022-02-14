# coding: UTF-8
import time, os
import torch
import numpy as np
from train_eval import train, test
from bert import Model, Config
import argparse
from utils import build_dataset, build_iterator, get_time_dif, load_dataset, gettoken
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertForMaskedLM, BertTokenizer, BertConfig
import torch.distributed as dist
from io import BytesIO
import torch
import oss2

parser = argparse.ArgumentParser(description='Salient triple classification')
parser.add_argument("--do_train", type=bool, default=True, help="Whether to run training.",)

parser.add_argument("--data_dir", default="data", type=str, help="The task data directory.")
parser.add_argument("--predict_path", default="data/", type=str, help="The task data directory.")
parser.add_argument("--model_dir", default="bert_pretrain/", type=str, help="The directory of pretrained models")
parser.add_argument("--output_dir", default='output/save_dict/', type=str, help="The path of result data and models to be saved.")
# models param
parser.add_argument("--max_length", default=256, type=int, help="the max length of sentence.")
parser.add_argument("--batch_size", default=8, type=int, help="Batch size for training.")
parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
parser.add_argument("--dropout", default=0.1, type=float, help="Drop out rate")
parser.add_argument("--epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument('--save_steps', type=int, default=1000, help="Save checkpoint every X updates steps.")
parser.add_argument('--seed', type=int, default=1, help="random seed for initialization")
parser.add_argument('--hidden_size', type=int, default=768,  help="random seed for initialization")

args = parser.parse_args()


def train_entry():
    start_time = time.time()
    print("Loading data...")
    train_data = load_dataset(config.train_path, config)
    dev_data = load_dataset(config.dev_path, config)
    test_data = load_dataset(config.test_path, config)
    # print(dev_data)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    #     train_data,
    #     shuffle=True)
    train_iter = DataLoader(
        train_data,
        shuffle=True,
        batch_size=config.batch_size,
        # sampler=train_sampler,
        num_workers=config.num_workers,
        drop_last=True)
    dev_iter = DataLoader(dev_data, shuffle=False, batch_size=config.batch_size,
                          num_workers=config.num_workers, drop_last=True)
    test_iter = DataLoader(test_data, shuffle=False, batch_size=config.batch_size,
                           num_workers=config.num_workers, drop_last=True)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    # train
    model = Model(config).to(config.device)

    train(config, model, train_iter, dev_iter, test_iter)


def test_entry():
    # test_data = load_online_data(config.predict_table, config)
    test_data = load_dataset(config.test_path, config)
    model = Model(config).to(config.device)

    model.load_state_dict(torch.load(config.save_path+"model.ckpt"))
    model.eval()
    loader = DataLoader(test_data, shuffle=False, batch_size=config.batch_size)
    pred, true, pred_true = 0, 0, 0
    loss_total = 0
    for b in range(5, 100, 5):
        a = b/100
        for i, batches in enumerate(loader):
            sent, _, _, labels = batches
            input_ids, attention_mask, type_ids = gettoken(config, sent)
            input_ids, attention_mask, type_ids, labels = \
                input_ids.to(config.device), attention_mask.to(config.device), type_ids.to(config.device), labels.to(
                    config.device)
            loss, pmi = model(input_ids, attention_mask, type_ids, labels)
            loss_total += loss.item()
            tmp_pred, tmp_true, tmp_pred_true = acc(pmi, labels, a)
            pred += tmp_pred
            true += tmp_true
            pred_true += tmp_pred_true

        if pred > 0 and pred_true > 0 and true > 0:
            print("pred", pred, true, pred_true)
            p = pred_true/pred
            r = pred_true/true
            f1 = 2 * p * r/(p + r)
            print(a, "f1:{},p:{},r,{}".format(f1, p, r))
        else:
            print("f1:{},p:{},r,{}".format(0, 0, 0))


def acc(logit, labels, b):
    logit = logit.cpu().detach().numpy()
    labels = labels.to('cpu').numpy()
    pred, true, pred_true = 0, 0, 0
    for logi, a in zip(logit, labels):
        # print(logi, a)
        if logi > b and a == 1:
            pred_true += 1
        if logi > b:
            pred += 1
        if a == 1:
            true += 1
    return pred, true, pred_true


if __name__ == '__main__':
    dataset = 'data'  # 数据集
    mode = "offline"
    config = Config(mode, args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    if not args.do_train:
        test_entry()
    else:
        train_entry()


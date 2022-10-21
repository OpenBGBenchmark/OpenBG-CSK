# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
import json
from utils import get_time_dif, gettoken, gettoken_pmi
from transformers import AdamW


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = 1e12
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, batches in enumerate(train_iter):
            model.zero_grad()
            # _, subjects, objects, predicates, labels = batches
            subjects, objects, predicates, labels = batches["subject"], batches["object"], batches["predicate"], batches["salience"]
            labels = [float(x) for x in labels]
            labels = np.asarray(labels)
            labels = torch.from_numpy(labels)
            if config.model == "PMI":
                input_ids, attention_mask, masked_head, masked_tail, masked_both = gettoken_pmi(config, subjects, objects, predicates)
                input_ids, masked_head, masked_tail, masked_both, labels, attention_mask = input_ids.to(
                    config.device), masked_head.to(config.device), masked_tail.to(config.device), masked_both.to(config.device), labels.to(config.device), attention_mask.to(config.device)
                pmi = model(input_ids, masked_head, masked_tail, masked_both, attention_mask)
                loss = F.mse_loss(pmi, labels.float())
            else:
                input_ids, attention_mask, type_ids, position_ids = gettoken(config, subjects, objects, predicates)
                input_ids, attention_mask, type_ids, labels = \
                    input_ids.to(config.device), attention_mask.to(config.device), type_ids.to(config.device), labels.to(config.device)
                position_ids = position_ids.to(config.device)
                pmi = model(input_ids, attention_mask, type_ids, position_ids)
                loss = F.binary_cross_entropy(pmi, labels.float(), reduction='sum')
            loss.backward()
            optimizer.step()
            total_batch += 1
            if total_batch % config.test_batch == 1:
                time_dif = get_time_dif(start_time)
                print("test:")
                f1, _, dev_loss, predict, ground, sents = evaluate(config, model, dev_iter, test=False)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Time: {2}'
                print(msg.format(total_batch, loss.item(), time_dif))
                print("loss", total_batch, loss.item(), dev_loss)
                if dev_loss < dev_best_loss:
                    print("save", dev_loss)
                    torch.save(model.state_dict(), config.save_path + "model.ckpt")
                    dev_best_loss = dev_loss
                model.train()

    test(config, model, test_iter)


def evaluate(config, model, data_iter, test=True):
    model.eval()
    loss_total = 0
    predicts, sents, grounds, all_bires = [], [], [], []
    with torch.no_grad():
        for i, batches in enumerate(data_iter):
            subjects,objects,predicates,labels = batches["subject"],batches["object"],batches["predicate"],batches["salience"]
            labels = [float(x) for x in labels]
            labels = np.asarray(labels)
            labels = torch.from_numpy(labels)
            if config.model == "PMI":
                input_ids, attention_mask, masked_head, masked_tail, masked_both = gettoken_pmi(config, subjects, objects, predicates)
                input_ids, masked_head, masked_tail, masked_both, labels, attention_mask = input_ids.to(
                    config.device), masked_head.to(config.device), masked_tail.to(
                    config.device), masked_both.to(config.device), labels.to(config.device), attention_mask.to(
                    config.device)
                pmi = model(input_ids, masked_head, masked_tail, masked_both, attention_mask)
                loss = F.mse_loss(pmi, labels.float())
            else:
                input_ids, attention_mask, type_ids, position_ids = gettoken(config, subjects, objects, predicates)
                input_ids, attention_mask, type_ids, labels = \
                    input_ids.to(config.device), attention_mask.to(config.device), type_ids.to(config.device), labels.to(config.device)
                position_ids = position_ids.to(config.device)
                pmi = model(input_ids, attention_mask, type_ids, position_ids)
                loss = F.binary_cross_entropy(pmi, labels.float(), reduction='sum')
            loss_total += loss.item()
            bires = torch.where(pmi > 0.5, torch.tensor([1]).to(config.device), torch.tensor([0]).to(config.device))
            for b, g, p in zip(bires, labels, pmi):
                all_bires.append(b.item())
                predicts.append(p.item())
                grounds.append(g.item())
    print("test set size:", len(grounds))
    accuracy = metrics.accuracy_score(grounds, all_bires)
    p = metrics.precision_score(grounds, all_bires, zero_division=0)
    r = metrics.recall_score(grounds, all_bires, zero_division=0)
    f1 = metrics.f1_score(grounds, all_bires, zero_division=0)
    print("f1:{},p:{},r,{}, accuracy:{}".format(f1, p, r, accuracy))
    return f1, pmi, loss_total / len(data_iter), predicts, grounds, sents


def predict(config, model, data_iter):
    model.eval()
    predicts = []
    with torch.no_grad():
        for i, batches in enumerate(data_iter):
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
            bires = torch.where(pmi > 0.2, torch.tensor([1]).to(config.device), torch.tensor([0]).to(config.device))
            for b, t in zip(bires, triple_id):
                predicts.append({"salience": b.item(), "triple_id": t})
    with open(config.save_path + "OpenBG-CSK_test.jsonl", "w") as f:
        for t in predicts:
            f.write(json.dumps(t, ensure_ascii=False)+"\n")


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path + "model.ckpt"))
    model.eval()
    start_time = time.time()
    predict(config, model, test_iter)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif, gettoken
from transformers import AdamW


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = 1000

    if config.mode == "online":
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, batches in enumerate(train_iter):
            model.zero_grad()
            sent, _, _, labels, _ = batches
            input_ids, attention_mask, type_ids, position_ids = gettoken(config,sent)
            input_ids, attention_mask, type_ids, labels = \
                input_ids.to(config.device), attention_mask.to(config.device), type_ids.to(config.device), labels.to(config.device)
            position_ids = position_ids.to(config.device)
            loss, pmi = model(input_ids, attention_mask, type_ids, position_ids, labels)
            loss.backward()
            optimizer.step()
            total_batch += 1
            if i % 200 == 1:
                time_dif = get_time_dif(start_time)
                print("test:")
                f1, _, dev_loss, wrong = evaluate(config, model, dev_iter, test=False)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Time: {2}'
                print(msg.format(total_batch, loss.item(), time_dif))
                print("loss", total_batch, loss.item(), dev_loss)
                if dev_loss < dev_best_loss:
                    print("save", dev_loss)
                    best_f1 = f1
                    dev_best_loss = dev_loss
                model.train()

    # test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path+"model.ckpt"))
    model.eval()
    start_time = time.time()
    _, _, test_loss,_ = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2}'
    print(msg.format(test_loss))
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=True):
    # model.eval()
    loss_total = 0
    pred, true, pred_true, correct_num, all = 0, 0, 0,0,0
    all_wrong = []
    final_score = 0
    with torch.no_grad():
        for i, batches in enumerate(data_iter):
            sent, _, _, labels, scores = batches
            input_ids, attention_mask, type_ids, position_ids = gettoken(config,sent)
            input_ids, attention_mask, type_ids, labels = \
                input_ids.to(config.device), attention_mask.to(config.device), type_ids.to(config.device), labels.to(
                    config.device)
            position_ids = position_ids.to(config.device)
            loss, pmi = model(input_ids, attention_mask, type_ids, position_ids, labels)
            loss_total += loss.item()
            tmp_pred, tmp_true, tmp_pred_true, wrong, score, tmp_correct, tmp_all = acc(sent, pmi, labels, scores)
            all_wrong.extend(wrong)
            final_score += score
            pred += tmp_pred
            true += tmp_true
            pred_true += tmp_pred_true
            correct_num += tmp_correct
            all += tmp_all
    if pred > 0:
        print("pred", pred, true, pred_true, correct_num, all)
        final_score = final_score/pred
        p = pred_true/pred
        r = pred_true/true
        if pred_true != 0:
            f1 = 2 * p * r/(p + r)
        else:
            f1 = 0
        print("f1:{},p:{},r,{}, final_score:{}, acc:{}".format(f1, p, r, final_score, correct_num/all))
    else:
        f1 = 0
        print("pred", 0, 0, 0, final_score)
    return f1, pmi, loss_total / len(data_iter), all_wrong


def acc(sent, logit, labels, scores):
    logit = logit.cpu().detach().numpy()
    labels = labels.to('cpu').numpy()
    pred, true, pred_true, correct_num, all = 0, 0, 0, 0, 0
    wrong = []
    final_score = 0
    for logi, a, s, c in zip(logit, labels, sent, scores):
        # print(logi, a)
        if logi > 0.8 and a == 1:
            pred_true += 1
        wrong.append(s+' '+str(a))
        if logi > 0.8:
            final_score += c
            pred += 1
        if a == 1:
            true += 1
        if (a == 1 and logi > 0.55) or (a == 0 and logi <= 0.55):
            correct_num += 1
        all += 1
    return pred, true, pred_true, wrong, final_score, correct_num, all

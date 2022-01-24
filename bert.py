# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from transformers import BertModel, BertTokenizer, BertPreTrainedModel, BertForMaskedLM
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch.nn.functional as F


class Config(object):

    """配置参数"""
    def __init__(self, dataset, mode):
        self.model_name = 'bert'
        self.mode = mode
        self.rank = -1
        self.local_rank = -1
        self.train_path = "data/train.csv"  # 训练集
        self.dev_path = 'data/dev.csv'  # 验证集
        self.test_path = 'data/test.csv'  # 测试集
        self.bert_path = 'data/roberta/'
        self.save_path = 'data/output/'
        if mode == "online":
            self.device = "cuda"
        else:
            self.device = "cpu"

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.num_workers = 4
        self.local_rank = -1
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 2                         # 类别数
        self.num_epochs = 10                                            # epoch数
        self.batch_size = 8                                           # mini-batch大小
        self.learning_rate = 1e-5                                       # 学习率
        self.weight_decay = 0.01
        self.dropout = 0.1
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path, do_lower_case=True)
        self.hidden_size = 768
        self.tagsize = 7
        self.max_length = 256
        self.adam_epsilon = 1e-8


class Model(nn.Module):
    def __init__(self, config, ):
        super(Model, self).__init__()
        self.num_labels = 1
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.dropout = nn.Dropout(config.dropout)
        self.dense_1 = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, type_ids, position_ids, labels):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=type_ids, position_ids=position_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        # [batch,hidden]
        procuct_ouput = torch.mean(sequence_output, 1)
        x = self.dense_1(procuct_ouput)
        x = torch.sigmoid(x).squeeze(-1)
        loss = F.binary_cross_entropy(x, labels.float(), reduction='sum')
        return loss, x


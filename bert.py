# coding: UTF-8
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertPreTrainedModel, BertForMaskedLM


class Config(object):

    """配置参数"""
    def __init__(self, args):
        self.model_name = 'bert'
        self.rank = -1
        self.local_rank = -1
        self.train_path = args.data_dir + '/train_triple.jsonl'  # 训练集
        self.test_path = args.data_dir + '/dev_triple.jsonl'  # 测试集
        self.save_path = args.output_dir  # 模型训练结果
        self.bert_path = args.model_dir
        self.test_batch = args.test_batch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.num_workers = 1
        self.local_rank = -1
        self.num_classes = 2                         # 类别数
        self.num_epochs = args.epochs                                            # epoch数
        self.batch_size = args.batch_size                                           # mini-batch大小
        self.learning_rate = args.learning_rate                                     # 学习率
        self.weight_decay = args.weight_decay
        self.dropout = args.dropout
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path, do_lower_case=True)
        self.hidden_size = args.hidden_size
        self.max_length = args.max_length


class Model(nn.Module):
    def __init__(self, config, ):
        super(Model, self).__init__()
        self.num_labels = 1
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.dropout = nn.Dropout(config.dropout)
        self.dense_1 = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, type_ids, position_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=type_ids, position_ids=position_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        procuct_ouput = torch.mean(sequence_output, 1)
        x = self.dense_1(procuct_ouput)
        x = torch.sigmoid(x).squeeze(-1)
        return x


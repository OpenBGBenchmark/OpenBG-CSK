# coding: UTF-8
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertPreTrainedModel, BertForMaskedLM, AutoModelForMaskedLM


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
        self.dropout = 0.1
        self.model=args.model


class KGBERT(nn.Module):
    def __init__(self, config, ):
        super(KGBERT, self).__init__()
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


class PMI(nn.Module):
    def __init__(self, config):
        super(PMI, self).__init__()
        self.device = config.device
        self.config = config
        self.bert = AutoModelForMaskedLM.from_pretrained(config.bert_path).to(self.device)

        # for param in self.bert.parameters():
        #     param.requires_grad = False

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.ReLU(),
            nn.Dropout(p=config.dropout),
            torch.nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.lstm_head = torch.nn.LSTM(input_size=config.hidden_size,
                                       hidden_size=config.hidden_size // 2,
                                       num_layers=2,
                                       dropout=config.dropout,
                                       bidirectional=True,
                                       batch_first=True)
        self.seq_indices = torch.LongTensor(list(range(config.new_tokens))).to(config.device)
        self.extra_token_embeddings = nn.Embedding(config.new_tokens, config.hidden_size).to(config.device)
        self.embeddings = self.bert.get_input_embeddings()
        self.lamda = torch.nn.Parameter(torch.tensor([config.lamda]))

        self.scale_a = torch.nn.Parameter(torch.tensor([0.66]))
        self.scale_b = torch.nn.Parameter(torch.tensor([0.66]))
        self.relu = torch.nn.ReLU()
        self.lamda = torch.nn.Parameter(torch.tensor([config.lamda]))

    def forward(self, sent, masked_head, masked_tail, masked_both, attention_mask):
        # def forward(self, masked, ids):
        # [5, 768]

        new_token_embeddings = self.extra_token_embeddings(self.seq_indices).unsqueeze(0)
        new_token_embeddings = self.mlp(self.lstm_head(new_token_embeddings)[0]).squeeze()
        # # [8,5,768]
        new_token_embeddings = new_token_embeddings.expand(min(self.config.batch_size, len(sent)), -1, -1)

        # [8, 64, 768]
        tail_embedding = self.embeddings(masked_tail)

        for bid in range(masked_tail.shape[0]):
            k = 0
            for cid in range(masked_tail[bid].shape[0]):
                if masked_tail[bid][cid] == 105:
                    tail_embedding[bid, cid, :] = new_token_embeddings[bid][k]
                    k += 1
                    if k >= self.config.new_tokens:
                        break

        head_embedding = self.embeddings(masked_head)
        for bid in range(masked_head.shape[0]):
            k = 0
            for cid in range(masked_head[bid].shape[0]):
                if masked_head[bid][cid] == 105:
                    head_embedding[bid, cid, :] = new_token_embeddings[bid][k]
                    k += 1
                    if k >= self.config.new_tokens:
                        break

        both_embedding = self.embeddings(masked_both)
        for bid in range(masked_both.shape[0]):
            k = 0
            for cid in range(masked_both[bid].shape[0]):
                if masked_both[bid][cid] == 105:
                    both_embedding[bid, cid, :] = new_token_embeddings[bid][k]
                    k += 1
                    if k >= self.config.new_tokens:
                        break

        # p(t|h,r)
        logprob_tail_conditional = self.predict(sent, tail_embedding, masked_tail, attention_mask)
        # p(h|t,r)
        logprob_head_conditional = self.predict(sent, head_embedding, masked_head, attention_mask)
        # marginal
        # p(t|r)
        logprob_tail_marginal = self.predict(sent, both_embedding, masked_tail, attention_mask)
        # p(h|r)
        logprob_head_marginal = self.predict(sent, both_embedding, masked_head, attention_mask)

        # NPMI average approximations of NPMI(t,h|r) and NPMI(h,t|r)
        mutual_inf_0 = (logprob_tail_conditional - self.scale_a * logprob_tail_marginal) / (
                -logprob_tail_conditional - self.scale_b * logprob_head_marginal)
        # Nec
        mutual_inf_1 = (logprob_head_conditional - self.scale_a * logprob_head_marginal) / (
                -logprob_head_conditional - self.scale_b * logprob_tail_marginal)
        mutual_inf = mutual_inf_0 * self.lamda + mutual_inf_1 * (1 - self.lamda)
        return mutual_inf

    def predict(self, sent, embeddings, masked, attention_mask):
        pred = self.bert(inputs_embeds=embeddings, attention_mask=attention_mask).logits
        pred = pred.log_softmax(2)
        logprob = torch.zeros(min(self.config.batch_size, len(sent))).to(self.device)
        # [8,69]
        for i, m in enumerate(masked):
            masked_ids = [idx for idx, token in enumerate(m) if token == 103]
            for idx in masked_ids:
                logprob[i] += pred[i, idx, sent[i][idx]]
            # logprob[i] = logprob[i]/len(masked_ids)
        return logprob



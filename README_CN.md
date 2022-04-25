<p align="left">
    <b> <a href="https://github.com/OpenBGBenchmark/OpenBG-CSK/blob/master/README.md">English</a> | 简体中文 </b>
</p>

# OpenBG-CSK
"CCKS2022 面向数字商务的知识图谱评测任务一：商品常识知识显著性推理"基线方法
# 使用
## 实验
运行以下指令，可以进行试验结果复现：
`python run.py` 
运行完成之后，F1分数，准确率和其他细节结果将会被输出.

#### 训练模型

运行shell文件: `bash run.sh`，该文件的内容如下：

```shell
DATA_DIR="data"

MODEL_DIR="bert-base-chinese"
OUTPUT_DIR="output/save_dict/"
PREDICT_DIR="data/"
MAX_LENGTH=128

python run.py \
    --data_dir=${DATA_DIR} \
    --model_dir=${MODEL_DIR} \
    --output_dir=${OUTPUT_DIR} \
    --do_train=True \
    --max_length=${MAX_LENGTH} \
    --batch_size=16 \
    --learning_rate=1e-5 \
    --epochs=10 \
    --seed=2021
```


#### 推理&得到结果

运行shell文件： `bash run.sh predict`，该文件的内容如下：
```shell
DATA_DIR="data"

MODEL_DIR="bert-base-chinese"
OUTPUT_DIR="output/save_dict/"
PREDICT_DIR="data/"
MAX_LENGTH=128

python run.py \
    --data_dir=${DATA_DIR} \
    --model_dir=${MODEL_DIR} \
    --output_dir=${OUTPUT_DIR} \
    --do_train=False \
    --max_length=${MAX_LENGTH} \
    --batch_size=16 \
    --learning_rate=1e-5 \
    --epochs=10 \
    --seed=2021
```

## 细节结果
数据集样例被放在`data`文件夹中：
`train_triple.jsonl`,`dev_triple.jsonl`. 

在`train_triple.jsonl`文件中，数据形式是`{"triple_id" : "0579","subject":"瓶装水","object":"跑步","predicate":"品类_适用_场景","salience": 0}`。

在`dev_triple.jsonl`文件中，数据形式是`{"triple_id":"0579","subject":"瓶装水","object":"跑步","predicate":"适用"}`。

# 基准
我们在实验数据集上测试了几个模型。我们使用KG-BERT作为基础模型并且报告了任务的基线结果如下。

| Model              | F1        | Acc.      |
| ------------------ | --------- | --------- |
| [BERT-base](https://huggingface.co/bert-base-chinese)          | 55.2 | 55.8 |
| [RoBERTa-wwm-ext-base](https://huggingface.co/hfl/chinese-roberta-wwm-ext)| 56.9 | 57.2|

KG-BERT将 头实体h，关系r，尾实体t 的文本作为双向编码器（例如bert）的输入，并且使用语言模型计算该三元组的得分。
具体而言，模型的输入是以下的格式，将h, r, t 拼接成 [CLS] h [SEP] r [SEP] t [SEP] 的格式。最终使用 权重W 对应的分类器对 [CLS] 相对的输出结果进行打分。

我们列出了在基线实验期间的超参数如下：

**常规超参数**

|       Param       | Value |
| :---------------: | :---: |
|   weight_decay    | 0.01  |
|   learning_rate   | 1e-5  |


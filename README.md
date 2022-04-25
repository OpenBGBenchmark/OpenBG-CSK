<p align="left">
    <b> English | <a href="https://github.com/OpenBGBenchmark/OpenBG-CSK/blob/master/README_CN.md">简体中文</a> </b>
</p>

# OpenBG-CSK
# Usage
## Experiments
To reproduce the experiments, run 
`python run.py` 
to run.

F1 score, accuracy and the detailed results will be printed.

#### Training models

Running shell files: `bash run.sh`, and the contents of shell files are as follow:

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


#### Inference & generation of results

Running shell files: `bash run.sh predict`, and the contents of shell files are as follows:
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

## Detailed Results
The sample of dataset are putted in the `data` repo:
`train_triple.jsonl`,`dev_triple.jsonl`. 

In `train_triple.jsonl`, the data format is `{"triple_id" : "0579","subject":"瓶装水","object":"跑步","predicate":"品类_适用_场景","salience": 0}`.

In `dev_triple.jsonl`, the data format is `{"triple_id":"0579","subject":"瓶装水","object":"跑步","predicate":"适用"}`.

# Benchmark
We evaluate the several models on the experiment dataset. We use KG-BERT as the base model and report the baselines of the task. 

| Model              | F1        | Acc.      |
| ------------------ | --------- | --------- |
| [BERT-base](https://huggingface.co/bert-base-chinese)          | 55.2 | 55.8 |
| [RoBERTa-wwm-ext-base](https://huggingface.co/hfl/chinese-roberta-wwm-ext)| 56.9 | 57.2|

KG-BERT takes texts of h, r, t as input of bidirectional encoder such as BERT and computes scoring function of the triple with language model. In specific, the input of model is the concatenation of h, r, t, as [CLS] h [SEP] r [SEP] t [SEP]. The final hidden state C corresponding to [CLS] and the classification layer weights W are used to calculate the triple score.

We list hyper-parameters during the baseline experiments.

**Common hyper-parameters**

|       Param       | Value |
| :---------------: | :---: |
|   weight_decay    | 0.01  |
|   learning_rate   | 1e-5  |


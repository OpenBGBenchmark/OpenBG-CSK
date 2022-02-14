# AliOpenKG-BSEE
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

MODEL_DIR="bert_pretrain"
OUTPUT_DIR="output/save_dict"
PREDICT_DIR="data/"
MAX_LENGTH=128

python run.py \
    --data_dir=${DATA_DIR} \
    --model_dir=${MODEL_DIR} \
    --output_dir=${OUTPUT_DIR} \
    --predict_path=${PREDICT_DIR} \
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

MODEL_DIR="bert_pretrain"
OUTPUT_DIR="output/save_dict"
PREDICT_DIR="data/"
MAX_LENGTH=128

python run.py \
    --data_dir=${DATA_DIR} \
    --model_dir=${MODEL_DIR} \
    --output_dir=${OUTPUT_DIR} \
    --predict_path=${PREDICT_DIR} \
    --do_train=False \
    --max_length=${MAX_LENGTH} \
    --batch_size=16 \
    --learning_rate=1e-5 \
    --epochs=10 \
    --seed=2021
```

## Detailed Results
The sample of dataset are putted in the `data` repo:
`train.tsv`,`test.tsv`. 

In `train.tsv`, the data format is head entity \t tail entity \t relation \t entity_id.

In `test.tsv`, the data format is head entity \t tail entity \t relation \t entity_id.

# Benchmark
We evaluate the several models on the experiment dataset. We use KG-BERT as the base model and report the baselines of the task. 

| Model              | F1        | Acc.      |
| ------------------ | --------- | --------- |
| [BERT-base](https://huggingface.co/bert-base-chinese)          | 50.6 | 56.8 |
| [RoBERTa-wwm-ext-base](https://huggingface.co/hfl/chinese-roberta-wwm-ext)| 52.1 | 53.2|

KG-BERT takes texts of h, r, t as input of bidirectional encoder such as BERT and computes scoring function of the triple with language model. In specific, the input of model is the concatenation of h, r, t, as [CLS] h [SEP] r [SEP] t [SEP]. The final hidden state C corresponding to [CLS] and the classification layer weights W are used to calculate the triple score.

We list hyper-parameters during the baseline experiments.

**Common hyper-parameters**

|       Param       | Value |
| :---------------: | :---: |
|   weight_decay    | 0.01  |
|   learning_rate   | 1e-5  |


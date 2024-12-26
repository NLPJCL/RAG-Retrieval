[English](./README.md) | [中文](./README_zh.md)

# Setting Up the Environment

```bash
conda create -n rag-retrieval python=3.8 && conda activate rag-retrieval
# To avoid compatibility issues between automatically installed torch and local CUDA, it is recommended to manually install a torch version compatible with your local CUDA before proceeding to the next step.
pip install -r requirements.txt 
```

| Requirement | Recommend |
|---------------|-----------------|
| accelerate    | 1.0.1           |
| deepspeed     | 0.15.4          |
| transformers  | 4.44.2          |

# Fine-tuning the Model

After installing the dependencies, we will demonstrate how to fine-tune the open-source ranking model (BAAI/bge-reranker-v2-m3) using our own data, or train a ranking model from scratch using BERT-like models (hfl/chinese-roberta-wwm-ext) and LLM-like models (Qwen/Qwen2.5-1.5B). Additionally, we support distilling the ranking capabilities of LLM-like models into smaller BERT models.

# Data Format

For ranking models, we support the following standard data format:
```
{"query": str, "pos": List[str], "neg":List[str], "pos_scores": List, "neg_scores": List}
```
- `pos` contains all positive samples for the query.(When distillation or training data is multi-level label, it can also be positive and negative samples)
- `neg` contains all negative samples for the query.
- `pos_scores` contains the scores corresponding to all positive sample documents for the query.(When distillation or data is multi-level label, it can also be the scores of positive and negative samples)
- `neg_scores` contains the scores corresponding to all negative sample documents for the query.

For the reranker tasks, the following types of data are supported for fine-tuning:

- Binary label data: When the relevance between query and doc in the labeled data is binary data, that is, label only exists in 0 and 1, you  refer to the [t2rank_100.jsonl](../../../example_data/t2rank_100.jsonl) file.
```
{"query": str, "pos": List[str], "neg": List[str]}
```
For this kind of data, we use `Binary Cross Entropy` for training. By default, we pair query and positive example with a score of 1; query and negative example with a score of 0. When predicting, the final prediction score of the model is the logit output by the model, which can be normalized to the range of 0-1 through sigmoid.

- Multi-level label data: When the relevance between query and doc in the labeled data is multi-level data, that is, the label is a multi-level label (may be equal to 0, 1, 2, etc.), the user can specify the level of relevance in pos_scores. At this time, the data set will automatically scale the discrete labels evenly to the 0-1 score range. For example, if there are three levels of labels (0, 1, 2) in the data set, then label 0: 0, label 1: 0.5, label 2: 1
```
{"query": str, "pos": List[str], "pos_scores": List[int|float]}
```
For this kind of data, users need to manually specify the max label and min label when setting the dataset parameters (the default value of max label is 1 and min label is 0 ). In training, we use  `MSE` or `Binary Cross Entropy` in the soft label for training.

- Distillation data: Users can directly use `pos` (including both positive and negative samples) and `pos_scores` to construct a dataset (`pos_scores` is a continuous score ranging from 0-1), please refer to the [t2rank_100.distill.standard.jsonl](../../../example_data/t2rank_100.distill.standard.jsonl) file.
```
{"query": str, "pos": List[str], "pos_scores": List[int|float]}
```
For this kind of data, we use mean square loss `MSE` or binary cross entropy loss `Binary Cross Entropy` in the soft label for training. You can find the code for using LLM for relevance scoring in the [examples/distill_llm_to_bert](../../../examples/distill_llm_to_bert) directory.



# Training

## Training BERT-like Models, fsdp(ddp)

```bash
CUDA_VISIBLE_DEVICES="0,1" nohup accelerate launch \
--config_file ../../../config/xlmroberta_default_config.yaml \
train_reranker.py \
--config config/training_bert.yaml \
>./logs/training_bert.log &
```

## Distilling BERT-like Models, fsdp(ddp)

```bash
CUDA_VISIBLE_DEVICES="0,1" nohup accelerate launch \
--config_file ../../../config/xlmroberta_default_config.yaml \
train_reranker.py \
--config config/distilling_bert.yaml \
>./logs/distilling_bert.log &
```

## Training LLM Models, deepspeed(zero1-2, not for zero3)

```bash
CUDA_VISIBLE_DEVICES="0,1" nohup accelerate launch \
--config_file ../../../config/deepspeed/deepspeed_zero1.yaml \
train_reranker.py \
--config config/training_llm.yaml \
>./logs/training_llm_deepspeed1.log &
```

**Parameter Explanation**

Multi-gpu training config_file:
- For BERT-like models, fsdp is used by default to support multi-GPU training. Here are examples of configuration files:
  - [default_fsdp](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/default_fsdp.yaml): Use this configuration file for training a ranking model from scratch based on hfl/chinese-roberta-wwm-ext.
  - [xlmroberta_default_config](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/xlmroberta_default_config.yaml): Use this configuration file for fine-tuning based on BAAI/bge-reranker-base, maidalun1020/bce-reranker-base_v1, or BAAI/bge-reranker-v2-m3, as they are all trained based on the multilingual XLMRoberta.

- For LLM-like models, it is recommended to use deepspeed to support multi-GPU training. Currently, only the training stages of zero1 and zero2 are supported. Below are examples of configuration files.
  - [deepspeed_zero1](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/deepspeed/deepspeed_zero1.yaml)
  - [deepspeed_zero2](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/deepspeed/deepspeed_zero2.yaml)

- Modifications for multi-GPU training configuration:
  - Change `CUDA_VISIBLE_DEVICES="0"` in the command
  - Modify the `num_processes` in the aforementioned configuration files to the number of GPUs you want to use.


Model-related:
- `model_name_or_path`: The name of the open-source reranker model or the local server location where it is downloaded. Examples: BAAI/bge-reranker-base, maidalun1020/bce-reranker-base_v1. Training from scratch is also possible, such as using BERT: hfl/chinese-roberta-wwm-ext and LLM: Qwen/Qwen2.5-1.5B.
- `model_type`: Currently supports bert_encoder or llm_decoder class models.
- `max_len`: The maximum input length supported by the model.

Dataset-related:
- `train_dataset`: The training dataset, format as described above.
- `val_dataset`: The validation dataset, same format as the training dataset.(If not, just set it to empty)
- `max_label`: The maximum label in the dataset, default is 1.
- `min_label`: The minimum label in the dataset, default is 0.

Training-related:
- `output_dir`: Directory for saving checkpoints and the final model during training.
- `loss_type`: Choose from point_ce (cross-entropy loss) and point_mse (mean squared error loss).
- `epoch`: Number of epochs to train the model on the dataset.
- `lr`: Learning rate, typically between 1e-5 and 5e-5.
- `batch_size`: Number of query-doc pairs in each batch.
- `seed`: Set a consistent seed for reproducibility of experimental results.
- `warmup_proportion`: Proportion of warmup steps to total model update steps. If set to 0, no warmup is performed, and cosine decay is applied directly from the set `lr`.
- `gradient_accumulation_steps`: Number of gradient accumulation steps. The actual batch size is `batch_size` * `gradient_accumulation_steps` * `num_of_GPUs`.
- `mixed_precision`: Whether to use mixed precision training to reduce GPU memory requirements. Mixed precision training optimizes memory usage by using low precision for computations and high precision for parameter updates. bf16 (Brain Floating Point 16) can effectively reduce anomalies in loss scaling but is only supported by some hardware.
- `save_on_epoch_end`: Whether to save the model at the end of each epoch.
- `num_max_checkpoints`: Controls the maximum number of checkpoints saved during a single training session.
- `log_interval`: Log loss every x parameter updates.
- `log_with`: Visualization tool, choose from wandb and tensorboard.

Model Parameters:
- `num_labels`: Number of logits output by the model, corresponding to the number of classification categories.
- For LLM used in discriminative ranking, the input format needs to be manually constructed, introducing the following parameters:
  - `query_format`, e.g., "query: {}"
  - `document_format`, e.g., "document: {}"
  - `seq`: Separates the query and document parts, e.g., " "
  - `special_token`: Indicates the end of the document content, prompting the model to start scoring. It can be any token, e.g., "\</s>"
  - Overall format: "query: xxx document: xxx\</s>"

# Loading the Model for Prediction

For saved models, you can easily load them for prediction. 

Cross-Encoder Models（BERT-like）
```python
ckpt_path = "./bge-reranker-m3-base"
reranker = CrossEncoder.from_pretrained(
    model_name_or_path=ckpt_path,
    num_labels=1,  # binary classification
)
reranker.model.to("cuda:0")
reranker.eval()

input_lst = [
    ["I love China", "I love China"],
    ["I love the USA", "I don't like the USA at all"],
]

res = reranker.compute_score(input_lst)

print(torch.sigmoid(res[0]))
print(torch.sigmoid(res[1]))
```
LLM-Decoder Models （MLP-based scalar mapping）
> To accommodate special cases of LLMs, such as "Qwen/Qwen2.5-1.5B," for discriminative ranking, a relevant format has been designed. The practical effect is: "query: {xxx} document: {xxx}\</s>". 
> 
> Experiments show that the introduction of `</s>` significantly enhances the ranking performance of LLMs [cited from https://arxiv.org/abs/2411.04539 section 4.3].

```python
ckpt_path = "./Qwen2-1.5B-Instruct"
reranker = LLMDecoder.from_pretrained(
    model_name_or_path=ckpt_path,
    num_labels=1,  # binary classification
    query_format="query: {}",
    document_format="document: {}",
    seq=" ",
    special_token="</s>",
)
reranker.model.to("cuda:0")
reranker.eval()

input_lst = [
    ["I love China", "I love China"],
    ["I love the USA", "I don't like the USA at all"],
]

res = reranker.compute_score(input_lst)

print(torch.sigmoid(res[0]))
print(torch.sigmoid(res[1]))
```
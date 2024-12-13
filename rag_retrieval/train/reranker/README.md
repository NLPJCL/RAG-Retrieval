[English](./README.md) | [中文](./README_zh.md)

# Setting Up the Environment

```bash
conda create -n rag-retrieval python=3.8 && conda activate rag-retrieval
# To avoid compatibility issues between automatically installed torch and local CUDA, it is recommended to manually install torch compatible with your local CUDA version before proceeding to the next step.
pip install -r requirements.txt 
```

# Fine-tuning the Model

After installing the dependencies, we will demonstrate how to fine-tune an open-source ranking model (BAAI/bge-reranker-v2-m3) using our own data, or train a ranking model from scratch using BERT-like models (hfl/chinese-roberta-wwm-ext) and LLM-like models (Qwen/Qwen2.5-1.5B). Additionally, we support distilling the ranking capabilities of LLM-like models into smaller BERT models.

# Data Format

For the ranking model, we support the following data format: Each line in the JSONL file is a dictionary string representing the distribution of positive and negative documents under a specific query.
```
{"query": str (required), "pos": List[str] (required), 
"neg":List[str](optional), 
"pos_scores": List(optional), "neg_scores": List(optional)}
```
- For binary classification data, where labels are either 0 or 1, refer to the [t2rank_100.jsonl](../../../example_data/t2rank_100.jsonl) file. During training, we use `Binary Cross Entropy loss` for optimization. We pair the query with positive examples (label=1) and the query with negative examples (label=0). In this case, "pos_scores" and "neg_scores" are not required. Therefore, during prediction, the final prediction score is the output logit of the model after applying sigmoid, ranging between 0 and 1.
- For multi-class classification data, where labels can be 0, 1, 2, ..., users can specify the maximum and minimum label values when setting up the dataset. The dataset will automatically scale the discrete label levels to the 0-1 score range. For example, label 0: 0, label 1: 0.5, label 2: 1.
- For continuous score data, such as scores from stronger model distillation, ranging from 0 to 1, refer to the [t2rank_100_distill.jsonl](../../../example_data/t2rank_100_distill.jsonl) file. For this type of data, we use `Binary Cross Entropy loss` or `Mean Squared Error (MSE) loss` for optimization under the context of soft labels. Code for scoring using LLM for distillation can be found in the examples/distill_llm_to_bert directory.

# Training

Run `bash train_reranker.sh` to start training. Below is the code executed by `train_reranker.sh`.

#BERT-like model, fsdp(ddp)

```bash
CUDA_VISIBLE_DEVICES="0"  nohup  accelerate launch --config_file ../../../config/xlmroberta_default_config.yaml train_reranker.py  \
--model_name_or_path "BAAI/bge-reranker-v2-m3" \
--train_dataset "../../../example_data/t2rank_100.jsonl" \
--val_dataset "../../../example_data/t2rank_100.jsonl" \
--output_dir "./output/t2ranking_100_example" \
--model_type "SeqClassificationRanker" \
--loss_type "point_ce" \
--batch_size 32 \
--lr 5e-5 \
--epochs 2 \
--num_labels 1 \
--log_with  'wandb' \
--save_on_epoch_end 1 \
--warmup_proportion 0.1 \
--gradient_accumulation_steps 3 \
--max_len 512 \
--gradient_accumulation_steps 4 \
--mixed_precision "bf16" \
 >./logs/t2ranking_100_example.log &
```

#LLM model, deepspeed(zero1-2, not for zero3)
```bash
CUDA_VISIBLE_DEVICES="4,5,6,7"  nohup  accelerate launch --config_file ../../../config/deepspeed/deepspeed_zero2.yaml train_reranker.py  \
--model_name_or_path "Qwen/Qwen2.5-1.5B" \
--train_dataset "../../../example_data/t2rank_100.jsonl" \
--val_dataset "../../../example_data/t2rank_100.jsonl" \
--output_dir "./output/t2ranking_100_example_llm_decoder" \
--model_type "SeqClassificationRanker" \
--loss_type "point_ce" \
--batch_size 8 \
--lr 5e-5 \
--epochs 2 \
--num_labels 1 \
--log_with 'wandb' \
--save_on_epoch_end 1 \
--warmup_proportion 0.1 \
--gradient_accumulation_steps 3 \
--max_len 512 \
--gradient_accumulation_steps 8 \
--mixed_precision "bf16" \
 >./logs/t2ranking_100_example_llm_decoder.log &
```

**Parameter Explanation**
- `model_name_or_path`: The name of the open-source reranker model or the local server path where it is downloaded. Examples: BAAI/bge-reranker-base, maidalun1020/bce-reranker-base_v1. You can also train from scratch, such as BERT: hfl/chinese-roberta-wwm-ext and LLM: Qwen/Qwen2.5-1.5B.
- `loss_type`: Choose between `point_ce` (Cross Entropy loss) and `point_mse` (Mean Squared Error loss).
- `model_type`: Currently supports SeqClassificationRanker.
- `save_on_epoch_end`: Whether to save the model at the end of each epoch.
- `log_with`: Visualization tool. If not set, default parameters will be used without errors.
- `batch_size`: Number of query-doc pairs in each batch.
- `lr`: Learning rate, typically between 1e-5 and 5e-5.

For BERT-like models, fsdp is used by default to support multi-GPU training. Below are example configuration files:
- [default_fsdp](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/default_fsdp.yaml): Use this configuration file for training from scratch based on hfl/chinese-roberta-wwm-ext.
- [xlmroberta_default_config](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/xlmroberta_default_config.yaml): Use this configuration file for fine-tuning based on BAAI/bge-reranker-base, maidalun1020/bce-reranker-base_v1, or BAAI/bge-reranker-v2-m3, as they are trained on the multilingual XLMRoberta.

For LLM-like models, it is recommended to use deepspeed to support multi-GPU training. Currently, only zero1 and zero2 training phases are supported. Below are example configuration files:
- [deepspeed_zero1](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/deepspeed/deepspeed_zero1.yaml)
- [deepspeed_zero2](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/deepspeed/deepspeed_zero2.yaml)

To modify the multi-GPU training configuration:
- Change the `CUDA_VISIBLE_DEVICES="0"` in train_reranker.sh to the desired multi-GPU setup.
- Modify the `num_processes` in the aforementioned configuration files to match the number of GPUs you want to use.

# Loading the Model for Prediction

You can easily load a saved model for prediction. An example is provided in `modeling.py`.

Noteably, to address special cases for discriminative ranking in Large Language Models (LLMs) like "Qwen/Qwen2.5-1.5B", specific formats are designed, effectively resulting in: "query: {xxx} document: {xxx}\<score>". Experiments show that the introduction of `special_token` significantly enhances LLM ranking performance. 

```python
ckpt_path = "maidalun1020/bce-reranker-base_v1" 
device = "cuda:0"
reranker = SeqClassificationRanker.from_pretrained(
    model_name_or_path=ckpt_path, 
    num_labels=1, # for binary classification
    cuda_device="cuda:0",
    loss_type="point_ce",
    query_format="{}",
    document_format="{}",
    seq="\n",
    special_token=""
)
# query_format="query: {}",
# document_format="document: {}",
# seq=" ",
# special_token="<score>"

reranker.eval()
reranker.model.to(device)

input_lst = [
    ["I love China", "I love China"],
    ["I love the United States", "I don't like the United States at all"],
    [
        "How long does it take to climb Mount Tai?",
        "It takes 1-8 hours to climb Mount Tai, depending on individual physical fitness. Professional climbers may only need a little over an hour to reach the summit, while those with lower physical fitness may take over 5 hours.",
    ],
]

res = reranker.compute_score(input_lst)

print(torch.sigmoid(res[0]))
print(torch.sigmoid(res[1]))
print(torch.sigmoid(res[2]))
```
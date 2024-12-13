
# 安装环境

```bash
conda create -n rag-retrieval python=3.8 && conda activate rag-retrieval
#为了避免自动安装的torch与本地的cuda不兼容，建议进行下一步之前先手动安装本地cuda版本兼容的torch。
pip install -r requirements.txt 
```

# 微调模型

在安装好依赖后，我们通过具体的示例来展示如何利用我们自己的数据来微调开源的排序模型 (BAAI/bge-reranker-v2-m3)，或者从 BERT 类模型 (hfl/chinese-roberta-wwm-ext) 以及 LLM 类模型 (Qwen/Qwen2.5-1.5B) 从零开始训练排序模型。与此同时，我们也支持将 LLM 类模型的排序能力蒸馏到较小的 BERT 模型中去。

# 数据格式

对于排序模型，我们支持如下数据格式：JSONL 文件中每一行是一个字典字符串，表示一个 query 下的文档正例负例分布情况。
```
{"query": str (required), "pos": List[str] (required), "neg":List[str](optional), "pos_scores": List(optional), "neg_scores": List(optional)}
```
- 当标注数据的相关性为二分类数据，即 label 只存在 0 和 1 时，可参考 [t2rank_100.jsonl](../../../example_data/t2rank_100.jsonl) 文件。
在训练中，我们采用二分类交叉熵损失 `Binary Cross Entropy`来进行优化。我们会把queyr和正例组成 pair，标签为 1，query和负例组成 pair，标签为 0。此时不用额外传输"pos_scores"和"neg_scores"。因此在预测时，模型最终的预测分数为模型输出的 logit 经过 sigmoid 后的值，范围在 0-1 之间。
- 当标注数据的相关性为多分类数据，即 label 可能等于 0,1,2,...。用户可以在设置数据集的时候指定 max label 和 min label 的大小。此时代码会自动将离散的 label 等级放缩到 0-1 分数区间中。例如 label 0: 0，label 1: 0.5，label 2: 1。
- 当标注数据的相关性为连续的分数时，即用其他模型进行蒸馏的分数，范围为 0-1，可参考 [t2rank_100_distill.jsonl](../../../example_data/t2rank_100_distill.jsonl) 文件。对于该种数据，在训练中，我们采用soft label 背景下的二分类交叉熵损失 `Binary Cross Entropy` 或均方损失 `MSE` 来进行优化。在examples/distill_llm_to_bert 目录下可以找到用 LLM 进行相关性打分标注的代码。



# 训练
执行 `bash train_reranker.sh` 即可，下面是 `train_reranker.sh` 执行的代码。

#bert类模型,fsdp(ddp)

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

#llm model, deepspeed(zero1-2, not for zero3)
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

**参数解释**
- model_name_or_path:开源的reranker模型的名称或下载下来的本地服务器位置。例如：BAAI/bge-reranker-base, maidalun1020/bce-reranker-base_v1，也可以从零开始训练，例如BERT: hfl/chinese-roberta-wwm-ext 和LLM: Qwen/Qwen2.5-1.5B）
- loss_type：可以在 point_ce（交叉熵损失） 和 point_mse（均方损失） 中选择
- model_type：当前支持 SeqClassificationRanker
- save_on_epoch_end：是否每个epoch都将模型存储下来。
- log_with：可视化工具，如果不设置用默认参数,也不会报错。
- batch_size：每个 batch 中 query-doc 的 pair 对的数量。
- lr：学习率，一般1e-5到5e-5之间。

对于 BERT 类模型，默认使用fsdp来支持多卡训练模型，以下是配置文件的示例。
- [default_fsdp](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/default_fsdp.yaml), 如果要在 hfl/chinese-roberta-wwm-ext 的基础上从零开始训练的排序，采用该配置文件。
- [xlmroberta_default_config](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/xlmroberta_default_config.yaml), 如果要在 BAAI/bge-reranker-base、maidalun1020/bce-reranker-base_v1、BAAI/bge-reranker-v2-m3 的基础上进行微调，采用该配置文件，因为其都是在多语言的 XLMRoberta 的基础上训练而来。


对于 LLM 类模型，建议使用 deepspeed 来支持多卡训练模型，目前只支持 zero1 和 zero2 的训练阶段，以下是配置文件的示例。
- [deepspeed_zero1](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/deepspeed/deepspeed_zero1.yaml)
- [deepspeed_zero2](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/deepspeed/deepspeed_zero2.yaml)

多卡训练配置文件修改:
- 修改 train_reranker.sh 的 CUDA_VISIBLE_DEVICES="0" 为你想要设置的多卡。
- 修改上述提到的配置文件的 num_processes 为你想要跑的卡的数量。


# 加载模型进行预测

对于保存的模型，你可以很容易加载模型来进行预测。在modeling.py中，我们给出了一个例子。

为了满足 LLM 如 "Qwen/Qwen2.5-1.5B" 用于判别式排序的特殊情况，设计了相关格式，实际效果为："query: {xxx} document: {xxx}<score>"，实验显示 special_token 的引入对 LLM 排序性能提升较大。

```python
ckpt_path = "maidalun1020/bce-reranker-base_v1" 
device = "cuda:0"
reranker = SeqClassificationRanker.from_pretrained(
    model_name_or_path=ckpt_path, 
    num_labels=1, # binary classification
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
    ["我喜欢中国", "我喜欢中国"],
    ["我喜欢美国", "我一点都不喜欢美国"],
    [
        "泰山要多长时间爬上去",
        "爬上泰山需要1-8个小时，具体的时间需要看个人的身体素质。专业登山运动员可能只需要1个多小时就可以登顶，有些身体素质比较低的，爬的慢的就需要5个多小时了。",
    ],
]

res = reranker.compute_score(input_lst)

print(torch.sigmoid(res[0]))
print(torch.sigmoid(res[1]))
print(torch.sigmoid(res[2]))
```


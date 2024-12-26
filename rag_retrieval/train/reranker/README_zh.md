
# 安装环境

```bash
conda create -n rag-retrieval python=3.8 && conda activate rag-retrieval
#为了避免自动安装的torch与本地的cuda不兼容，建议进行下一步之前先手动安装本地cuda版本兼容的torch。
pip install -r requirements.txt 
```

| Requirement | Recommend |
| ---------------| ---------------- |
| accelerate    |             1.0.1 |
|deepspeed |0.15.4|
|transformers |4.44.2|          


               

# 微调模型

在安装好依赖后，我们通过具体的示例来展示如何利用我们自己的数据来微调开源的排序模型 (BAAI/bge-reranker-v2-m3)，或者从 BERT 类模型 (hfl/chinese-roberta-wwm-ext) 以及 LLM 类模型 (Qwen/Qwen2.5-1.5B) 从零开始训练排序模型。与此同时，我们也支持将 LLM 类模型的排序能力蒸馏到较小的 BERT 模型中去。

# 数据格式

对于排序模型，我们支持如下的标准数据格式：
```
{"query": str, "pos": List[str], "neg":List[str], "pos_scores": List, "neg_scores": List}
```
- `pos` 为 query 下所有的正样本。(当蒸馏或者训练数据是多级标签时，也可以是正负样本)
- `neg` 为 query 下所有的负样本。
- `pos_scores` 为 query 下所有正样本对应的得分。(当蒸馏或者数据是多级标签时，也可以是正负样本的得分)
- `neg_scores` 为 query 下所有负样本对应的得分。


对于排序模型，支持以下几种数据进行微调：

- 二分类数据：当标注数据中query和doc的相关性为二分类数据，即 label 只存在 0 和 1 时，可参考 [t2rank_100.jsonl](../../../example_data/t2rank_100.jsonl) 文件。
```
{"query": str, "pos": List[str], "neg":List[str]}
```
对于这种数据，在训练中，我们采用二分类交叉熵损失 `Binary Cross Entropy`来进行训练。在默认情况下，我们会把 query 和正例组成 pair，分数为 1；query 和负例组成 pair，分数为 0。在预测时，模型最终的预测分数为模型输出的 logit，后续可以经过 sigmoid 归一化为 0-1 区间。

- 多级标签数据：当标注数据中query和doc的相关性为多分类数据，即 label 为多级标签，（可能等于 0,1,2 等）,用户可以在pos_scores中指定相关性的级别。此时数据集内部会自动将离散的 label 均匀放缩到 0-1 分数区间中。例如数据集中存在三级标签（0，1，2），那么 label 0: 0，label 1: 0.5，label 2: 1
```
{"query": str, "pos": List[str], "pos_scores": List[int|float]}
```
对于这种数据，用户在设置数据集参数的时候需要手动指定 max label 和 min label（初始条件下 max label 默认为 1，min label 默认为 0）。在训练中，我们采用均方损失 `MSE` 或者soft label 下的二分类交叉熵损失 `Binary Cross Entropy`来进行训练。

- 蒸馏数据：用户可以直接使用 `pos`（同时包含正样本和负样本）和 `pos_scores` 来构建数据集(`pos_scores` 为范围 0-1 的连续分数)，可参考 [t2rank_100.distill.standard.jsonl](../../../example_data/t2rank_100.distill.standard.jsonl) 文件。
```
{"query": str, "pos": List[str], "pos_scores": List[int|float]}
```
对于这种数据,在训练中，我们采用均方损失 `MSE` 或者soft label 下的二分类交叉熵损失 `Binary Cross Entropy`来进行训练。在 [examples/distill_llm_to_bert](../../../examples/distill_llm_to_bert) 目录下可以找到用 LLM 进行相关性打分标注的代码。


# 训练

#bert类模型训练, fsdp(ddp)

```bash
CUDA_VISIBLE_DEVICES="0,1" nohup accelerate launch \
--config_file ../../../config/xlmroberta_default_config.yaml \
train_reranker.py \
--config config/training_bert.yaml \
>./logs/training_bert.log &
```

#bert类模型蒸馏, fsdp(ddp)

```bash
CUDA_VISIBLE_DEVICES="0,1" nohup accelerate launch \
--config_file ../../../config/xlmroberta_default_config.yaml \
train_reranker.py \
--config config/distilling_bert.yaml \
>./logs/distilling_bert.log &
```

#llm model, deepspeed(zero1-2, not for zero3)
```bash
CUDA_VISIBLE_DEVICES="0,1" nohup accelerate launch \
--config_file ../../../config/deepspeed/deepspeed_zero1.yaml \
train_reranker.py \
--config config/training_llm.yaml \
>./logs/training_llm_deepspeed1.log &
```

**参数解释**

多卡训练config_file:

- 对于 BERT 类模型，默认使用fsdp来支持多卡训练模型，以下是配置文件的示例。
  - [default_fsdp](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/default_fsdp.yaml), 如果要在 hfl/chinese-roberta-wwm-ext 的基础上从零开始训练的排序，采用该配置文件
  -  [xlmroberta_default_config](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/xlmroberta_default_config.yaml), 如果要在 BAAI/bge-reranker-base、maidalun1020/bce-reranker-base_v1、BAAI/bge-reranker-v2-m3 的基础上进行微调，采用该配置文件，因为其都是在多语言的 XLMRoberta 的基础上训练而来

- 对于 LLM 类模型，建议使用 deepspeed 来支持多卡训练模型，目前只支持 zero1 和 zero2 的训练阶段，以下是配置文件的示例
  - [deepspeed_zero1](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/deepspeed/deepspeed_zero1.yaml)
  - [deepspeed_zero2](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/deepspeed/deepspeed_zero2.yaml)

- 多卡训练配置文件修改:
  - 修改命令中的 CUDA_VISIBLE_DEVICES="0" 为你想要设置的多卡
  - 修改上述提到的配置文件的 num_processes 为你想要跑的卡的数量


模型方面：
- `model_name_or_path`：开源的reranker模型的名称或下载下来的本地服务器位置。例如：BAAI/bge-reranker-base, maidalun1020/bce-reranker-base_v1，也可以从零开始训练，例如BERT: hfl/chinese-roberta-wwm-ext 和LLM: Qwen/Qwen2.5-1.5B）
- `model_type`：当前支持 bert_encoder或llm_decoder类模型
- `max_len`：数据支持的最大输入长度

数据集方面：
- `train_dataset`：训练数据集，格式见上文
- `val_dataset`：验证数据集，格式同训练集(如果没有，设置为空即可)
- `max_label`：数据集中的最大 label，默认为 1
- `min_label`：数据集中的最小 label，默认为 0

训练方面：
- `output_dir`：训练过程中保存的 checkpoint 和最终模型的目录
- `loss_type`：从 point_ce（交叉熵损失）和 point_mse（均方损失） 中选择
- `epoch`：模型在训练数据集上训练的轮数
- `lr`：学习率，一般1e-5到5e-5之间
- `batch_size`：每个 batch 中 query-doc pair 对的数量
- `seed`：设置统一种子，用于实验结果的复现
- `warmup_proportion`：学习率预热步数占模型更新总步数的比例，如果设置为 0，那么不进行学习率预热，直接从设置的 `lr` 进行余弦衰退
- `gradient_accumulation_steps`：梯度累积步数，模型实际的 batch_size 大小等于 `batch_size` * `gradient_accumulation_steps` * `num_of_GPUs`
- `mixed_precision`：是否进行混合精度的训练，以降低显存的需求。混合精度训练通过在计算使用低精度，更新参数用高精度，来优化显存占用。并且 bf16（Brain Floating Point 16）可以有效降低 loss scaling 的异常情况，但该类型仅被部分硬件支持
- `save_on_epoch_end`：是否在每一个 epoch 结束后都保存模型
- `num_max_checkpoints`：控制单次训练下保存的最多 checkpoints 数目
- `log_interval`：模型每更新 x 次参数记录一次 loss
- `log_with`：可视化工具，从 wandb 和 tensorboard 中选择

模型参数：
- `num_labels`：模型输出 logit 的数目，即为模型分类类别的个数
- 对于 LLM 用于判别式排序打分时，需要人工构造输入格式，由此引入下列参数
  - `query_format`, e.g. "query: {}"
  - `document_format`, e.g. "document: {}" 
  - `seq`：分隔 query 和 document 部分, e.g. " "
  - `special_token`：预示着 document 内容的结束，引导模型开始打分，理论上可以是任何 token, e.g. "\</s>" 
  - 整体的格式为："query: xxx document: xxx\</s>" 


# 加载模型进行预测

对于保存的模型，你可以很容易加载模型来进行预测。

Cross-Encoder 模型（BERT-like）
```python
ckpt_path = "./bge-reranker-m3-base"
reranker = CrossEncoder.from_pretrained(
    model_name_or_path=ckpt_path,
    num_labels=1,  # binary classification
)
reranker.model.to("cuda:0")
reranker.eval()

input_lst = [
    ["我喜欢中国", "我喜欢中国"],
    ["我喜欢美国", "我一点都不喜欢美国"]
]

res = reranker.compute_score(input_lst)

print(torch.sigmoid(res[0]))
print(torch.sigmoid(res[1]))
```

LLM-Decoder 模型 （基于 MLP 进行标量映射）

> 为了满足 LLM 如 "Qwen/Qwen2.5-1.5B" 用于判别式排序的特殊情况，设计了相关格式，实际效果为："query: {xxx} document: {xxx}\</s>"，实验显示 \</s> 的引入对 LLM 排序性能提升较大 [源于 https://arxiv.org/abs/2411.04539 section 4.3]。

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
    ["我喜欢中国", "我喜欢中国"],
    ["我喜欢美国", "我一点都不喜欢美国"],
]

res = reranker.compute_score(input_lst)

print(torch.sigmoid(res[0]))
print(torch.sigmoid(res[1]))
```


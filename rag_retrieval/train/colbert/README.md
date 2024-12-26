
# 安装环境
```bash
conda create -n rag-retrieval python=3.8 && conda activate rag-retrieval
#为了避免自动安装的torch与本地的cuda不兼容，建议进行下一步之前先手动安装本地cuda版本兼容的torch。
pip install -r requirements.txt 
```

# 微调模型
在安装好依赖后，我们通过具体的示例来展示如何利用我们自己的数据来微调开源的colbert模型。(bge-m3-colbert),或者你可以从BERT类模型(chinese-roberta-wwm-ext)开始从零训练自己的colbert模型。

# 数据格式


和bge类似，训练数据是一个jsonl文件，文件中每一行如下面的示例所示。其中pos是一组正例doc的文本，neg是一组负例doc的文本。

对于colbert模型，支持使用下面的数据进行微调：

- query和正例doc和难负例doc。此时负例为query对应的难负例，以及batch内随机负例,可以参考[example_data](https://github.com/NLPJCL/RAG-Retrieval/blob/master/example_data/t2rank_100.jsonl)文件。
```
{"query": str, "pos": List[str], "neg":List[str]}
```



# 训练
执行bash train_colbert.sh即可，下面是train_colbert.sh执行的代码。

```bash
CUDA_VISIBLE_DEVICES="0,1"   nohup  accelerate launch --config_file ../../../config/default_fsdp.yaml train_colbert.py  \
--model_name_or_path "hfl/chinese-roberta-wwm-ext" \
--dataset "../../../example_data/t2rank_100.jsonl" \
--output_dir "./output/t2ranking_100_example" \
--batch_size 4 \
--lr 5e-6 \
--epochs 2 \
--save_on_epoch_end 1 \
--gradient_accumulation_steps 12 \
--temperature 0.02 \
--query_max_len 128 \
--passage_max_len 512 \
--colbert_dim 768 \
--neg_nums 15 \
--log_with 'wandb' \
--warmup_proportion 0.1 \
 >./logs/t2ranking_100_example.log &
```

**参数解释**
- model_name_or_path : 开源的embedding模型的名称或下载下来的服务器位置.（可以是：BAAI/bge-m3,也可以从普通的bert类模型开始训练，例如hfl/chinese-roberta-wwm-ext）
- save_on_epoch_end : 是否每个epoch都将模型存储下来。
- log_with : 可视化工具，如果不设置用默认参数,也不会报错。
- batch_size : 每个batch query-doc的piar对的数量。
- colbert_dim :存储的维度。beg-m3-colbert是1024维度。
- neg_nums：训练过程中难负例的数量，其不应该超过训练数据集中真实负例neg:List[str]的个数，如果neg_nums大于真实的负例的个数，那么就会对真实负例进行重采样，再随机选取neg_nums个负例。如果neg_nums小于真实负例的个数，那么在训练过程中会随机选择neg_nums个。
- lr :学习率，bge-m3-colbert的学习率建议在ne-6级别。

默认使用fsdp来支持多卡训练模型，以下是配置文件的示例
- [default_fsdp](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/default_fsdp.yaml)。如果要在hfl/chinese-roberta-wwm-ext的基础上从零开始训练的排序，采用该配置文件。
- [xlmroberta_default_config](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/xlmroberta_default_config.yaml),如果要在BAAI/bge-m3的基础上进行微调，采用该配置文件，因为其在多语言的xlmroberta的基础上训练而来。

多卡训练配置文件修改:
- 修改train_colbert.sh的CUDA_VISIBLE_DEVICES="0"为你想要设置的多卡。
- 修改上述提到的配置文件的num_processes为你想要跑的卡的数量。


# 加载模型进行预测

对于保存的模型，你可以很容易加载模型来进行预测。在model.py里，我们给了一个示例如何加载以及预测。


```python

ckpt_path='BAAI/bge-m3'
device = 'cuda:0'
colbert = ColBERT.from_pretrained(
    ckpt_path,
    colbert_dim=1024,
    cuda_device=device
    )
colbert.eval()
colbert.to(device)
input_lst=[
    ['我喜欢中国','我喜欢中国'],
    ['我喜欢中国','我一点都不喜欢中国'],
    ['泰山要多长时间爬上去','爬上泰山需要1-8个小时，具体的时间需要看个人的身体素质。专业登山运动员可能只需要1个多小时就可以登顶，有些身体素质比较低的，爬的慢的就需要5个多小时了。']]

# input_lst=[['What is BGE M3?','BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.']]

res=colbert.compute_score(input_lst)
print(res[0])
print(res[1])
print(res[3])


```


# 微调模型
在安装好依赖后，我们通过具体的示例来展示如何利用我们自己的数据来微调开源的向量模型。

# 数据格式

和bge类似，训练数据是一个json文件，文件中每一行如下面的示例所示。其中pos是一组正例doc的文本，neg是一组负例doc的文本。

对于向量模型，支持以下两种数据进行微调：

- query和正例doc，此时负例为batch内随机负例。
```
{"query": str, "pos": List[str]}
```
- query和正例doc和难负例doc。此时负例为query对应的难负例，以及batch内随机负例,可以参考[example_data](https://github.com/NLPJCL/RAG-Retrieval/blob/master/example_data/t2rank_100.json)文件。
```
{"query": str, "pos": List[str], "neg":List[str]}
```

# 训练

```bash
CUDA_VISIBLE_DEVICES="0"   nohup  accelerate launch --config_file ../../config/default_fsdp.yaml train_embedding.py  \
--model_name_or_path "BAAI/bge-base-zh-v1.5" \
--dataset "../../example_data/t2rank_100.json" \
--output_dir "./output/t2ranking_100_example" \
--batch_size 4 \
--lr 2e-5 \
--epochs 2 \
--save_on_epoch_end 1 \
--gradient_accumulation_steps 24  \
--log_with 'wandb' \
--warmup_proportion 0.1 \
--neg_nums 15 \
--temperature 0.02 \
--query_max_len 128 \
--passage_max_len 512 \
 >./logs/t2ranking_100_example.log &
```

**参数解释**
- model_name_or_path:开源的embedding模型的名称或下载下来的服务器位置.（只要其支持sentence-transformers来进行推理，就可以来进行微调。）
- save_on_epoch_end:是否每个epoch都将模型存储下来。
- log_with：可视化工具，如果不设置用默认参数,也不会报错。
- neg_nums：训练过程中难负例的数量，其不应该超过训练数据集中真实负例neg:List[str]的个数，如果neg_nums大于真实的负例的个数，那么就会对真实负例进行重采样，再随机选取neg_nums个负例。如果neg_nums小于真实负例的个数，那么在训练过程中会随机选择neg_nums个。
- batch_size :越大batch内随机负例越多，效果越好。实际的负例个数为，(1+batch_size*neg_nums)
- lr :学习率，一般1e-5到5e-5之间。

默认使用fsdp来支持多卡训练模型，这里是一个配置文件的示例[default_fsdp](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/default_fsdp.yaml)。


# 加载模型进行预测

对于保存的模型，会按照sentence-transformers的格式去保存，因此你可以很容易加载模型来进行预测。

在model.py里，我们给了一个示例如何加载以及预测。


```python
cuda_device='cuda:0'
ckpt_path = '保存模型的路径'
embedding = Embedding.from_pretrained(
    ckpt_path,
)
embedding.to(cuda_device)
input_lst = ['我喜欢中国','我爱爬泰山']
embeddings = embedding.encode(input_lst,device=cuda_device)

```

同时，也支持使用sentence-transformers库来进行推理。
```python
from sentence_transformers import SentenceTransformer

ckpt_path = '保存模型的路径'
input_lst = ['我喜欢中国','我爱爬泰山']

model = SentenceTransformer(ckpt_path)
embeddings = model.encode(input_lst)

```

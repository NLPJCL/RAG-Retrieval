
# 安装环境
```bash
conda create -n rag-retrieval python=3.8 && conda activate rag-retrieval
#为了避免自动安装的torch与本地的cuda不兼容，建议进行下一步之前先手动安装本地cuda版本兼容的torch。
pip install -r requirements.txt 
```

# 微调模型
在安装好依赖后，我们通过具体的示例来展示如何利用我们自己的数据来微调开源的向量模型。

# 数据格式

和bge类似，训练数据是一个jsonl文件，文件中每一行如下面的示例所示。其中`pos`是一组正例doc的文本，`neg`是一组负例doc的文本,`prompt_for_query`是某些向量模型需要在query前面添加的prompt。（可选的）。

对于向量模型，支持以下四种数据进行微调：

- query和正例doc，此时负例为batch内随机负例（即其它query的正例doc作为负例）。
```
{"query": str, "pos": List[str], "prompt_for_query"(optional): str}
```
- query和正例doc和难负例doc。此时负例为query对应的难负例，以及batch内随机负例（即其它query的负例doc作为负例）,可以参考[example_data](https://github.com/NLPJCL/RAG-Retrieval/blob/master/example_data/t2rank_100.jsonl)文件。
```
{"query": str, "pos": List[str], "neg":List[str], "prompt_for_query"(optional): str}
```
- query和doc，以及query和每个doc的监督分数。可以参考[example_data](https://github.com/NLPJCL/RAG-Retrieval/blob/master/example_data/lmsft.jsonl)文件。监督信号的构建推荐两种方式：
  - 人工标注：类似[STS任务](https://huggingface.co/datasets/PhilipMay/stsb_multi_mt)，给query和每个文档根据相似度打分。
  - LLM标注：参考论文[Atlas](https://www.jmlr.org/papers/v24/23-0037.html) ，使用LLM的困惑度，或Encoder-Decoder架构Transformer的FiD分数。
```
{"query": str, "pos": List[str], "scores":List[float], "prompt_for_query"(optional): str}
```

- query和对应的teacher embedding。方法介绍：[infgrad/jasper_en_vision_language_v1](https://huggingface.co/infgrad/jasper_en_vision_language_v1)，蒸馏数据的构造，参考：[examples/stella_embedding_distill](../../../examples/stella_embedding_distill/)
   - text: 如下示例。
   - teacher_embedding: 一个np.memmap file.
```
{"query": str, "prompt_for_query"(optional): str}
``` 


注："prompt_for_query" 可用于将指令信息融入到 query 中，例如"Instruct: 给定一个用户问题, 检索出对回答问题有帮助的文档片段\nQuery: "。
# 训练

执行bash train_embedding.sh即可，下面是train_embedding.sh执行的代码。

#bert类模型,fsdp(ddp)
```bash
 CUDA_VISIBLE_DEVICES="0,1"   nohup  accelerate launch \
 --config_file ../../../config/default_fsdp.yaml \
 train_embedding.py  \
 --config ./config/training_embedding.yaml  \
 >./logs/t2ranking_100_example_bert.log &
```

#llm类模型,deepspeed(zero1-3)
```bash
 CUDA_VISIBLE_DEVICES="0,1"   nohup  accelerate launch \
 --config_file ../../../config/deepspeed/deepspeed_zero2.yaml \
 train_embedding.py  \
 --config ./config/training_embedding.yaml  \
 >./logs/t2ranking_100_example_llm.log &
```

#distill teacher embedding, fsdp(ddp) 参考：[infgrad/jasper_en_vision_language_v1](https://huggingface.co/infgrad/jasper_en_vision_language_v1)。
```bash
 CUDA_VISIBLE_DEVICES="0,1"   nohup  accelerate launch \
 --config_file ../../../config/default_fsdp.yaml \
 train_embedding.py  \
 --config ./config/distill_embedding.yaml  \
 >./logs/t2ranking_100_example_bert_distill.log &
```

**参数解释**
- model_name_or_path:开源的embedding模型的名称或下载下来的服务器位置.（只要其支持sentence-transformers来进行推理，就可以来进行微调。）
- save_on_epoch_end:是否每个epoch都将模型存储下来。
- log_with：可视化工具，如果不设置用默认参数,也不会报错。
- neg_nums：训练过程中难负例的数量，其不应该超过训练数据集中真实负例neg:List[str]的个数，如果neg_nums大于真实的负例的个数，那么就会对真实负例进行重采样，再随机选取neg_nums个负例。如果neg_nums小于真实负例的个数，那么在训练过程中会随机选择neg_nums个。(如果只有query和正例doc，可忽略该参数)
- batch_size :越大batch内随机负例越多，效果越好。实际的负例个数为: batch_size*neg_nums
- lr :学习率，一般1e-5到5e-5之间。
- use_mrl ：是否使用mlr训练。（新增加一个线性层[hidden_size,max(mrl_dims)]来进行MRL训练，参考[Matryoshka Representation Learning
](https://arxiv.org/abs/2205.13147)，默认使用MRL-E）
- mrl_dims ：要训练的mrl维度列表,例如，"128, 256, 512, 768, 1024, 1280, 1536, 1792"。



对于bert类模型，默认使用fsdp来支持多卡训练模型，以下是配置文件的示例：
- [default_fsdp](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/default_fsdp.yaml)。如果要在chinese-roberta-wwm-ext的基础上从零开始训练，采用该配置文件。
- [xlmroberta_default_config](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/xlmroberta_default_config.yaml),如果要在bge-m3-embedding和bce-embedding-base_v1的基础上进行微调，采用该配置文件，因为两者在多语言的xlmroberta的基础上训练而来。

对于llm类模型，默认使用deepspeed来支持多卡训练模型，以下是配置文件的示例：
- [deepspeed_zero2](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/deepspeed/deepspeed_zero2.yaml)

多卡训练配置文件修改:
- 修改train_embedding.sh的CUDA_VISIBLE_DEVICES="0"为你想要设置的多卡。
- 修改上述提到的配置文件的num_processes为你想要跑的卡的数量。

# 加载模型进行预测

对于保存的模型，会按照sentence-transformers的格式去保存，因此你可以很容易加载模型来进行预测。

在model.py里，我们给了一个示例如何加载以及预测。


```python
cuda_device = 'cuda:0'
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

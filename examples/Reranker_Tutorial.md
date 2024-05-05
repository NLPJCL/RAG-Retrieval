
欢迎使用rag_retrieval库的Reranker模块，这里是一份Reranker的Tutorial,主要来介绍下Reranker的功能以及注意事项，希望您可以使用的更加得心应手。


# 安装

```bash
#为了避免自动安装的torch与本地的cuda不兼容，建议进行下一步之前先手动安装本地cuda版本兼容的torch。
pip install rag-retrieval
```

# 整体功能
rag_retrieval的Reranker,支持以下的功能。


我们开发了一个轻量级的python库[rag-retrieval](https://pypi.org/project/rag-retrieval/),提供统一的方式调用任意不同的RAG排序模型，其有以下的特点。

1.支持多种排序模型：支持常见的开源排序模型(corss encoder reranker,decoder-only 的llm reranker)

2.长doc友好：支持两种不同的对于长doc的处理逻辑(最大长度截断，最大分值切分)。

3.益于扩展：如果有新的排序模型，用户只需要继承basereranker，并且实现rank以及comput_score函数即可。



下面介绍下加载模型的过程。


# 加载模型

```python

import os
os.environ['CUDA_VISIBLE_DEVICES']='7'

from rag_retrieval import Reranker

#如果自动下载对应的模型失败，请先从huggface下载对应的模型到本地，然后这里输入本地的路径。

ranker = Reranker('BAAI/bge-reranker-base',dtype='fp16',verbose=0)
```

推荐使用os.environ['CUDA_VISIBLE_DEVICES']来指定显存。

**Reranker的参数**

```python
Reranker的
-   Reranker的参数为：
    - model_name: str,
    - model_type: Optional[str] = None,
    - verbose: int = 1,
    - **kwargs
```

**参数解释**


- model_name: 为对应的模型名字或者模型在本地的路径，如果自动下载对应的模型失败，请先从huggface下载对应的模型到本地，然后这里输入本地的路径。
- model_type: 为对应的reranker模型类型，目前支持cross-encoder,llm,colbert，可以不具体指定，那么，代码会根据输入的model_name自动推断选择哪种类型。
- verbose，是否打印出必要的debug信息，默认打印，如果测试无误，可设置verbose=0不打印。
- **kwargs**: 可以在此指定一些模型相关的参数，例如：
    - device：推理设备，可以设置为'cpu','cuda'等。如果不具体指定，那么按照以下优先级使用。如果有gpu，默认使用gpu，有mps,默认mps,如果有npu,默认使用npu。否则，使用cpu推理。
    - dtype：加载模型的类型，可以设置为'fp32',fp16'，'bf16'。如果不具体指定，默认使用fp32。设置fp16可加快推理速度。


# 支持的reranker模型

## Cross Encoder ranker

对于cross encoder 的ranker，rag_retrieval的Reranker支持多个强大的开源模型,总的来说，只要其cross encoder是使用transformers的**AutoModelForSequenceClassification**的模型结构，那么就可以支持使用Reranker来进行推理。举例如下。

- **bge系列的cross encoder模型，例如(BAAI/bge-reranker-base, BAAI/bge-reranker-large, BAAI/bge-reranker-v2-m3 )**

- **bce的cross encoder模型，例如(maidalun1020/bce-reranker-base_v1)**


## LLM ranker 

对于LLM ranker，rag_retrieval的Reranker支持多种强大的定制化LLM排序模型。也支持使用任意的LLM的chat模型来进行zero shot排序。举例如下。

- **bge系列的llm ranker模型，例如(BAAI/bge-reranker-v2-gemma, BAAI/bge-reranker-v2-minicpm-layerwise, BAAI/bge-reranker-v2-m3 )**

- **也支持使用任意的LLM的chat模型来进行zero shot排序**


**下面介绍下Reranker返回的reranker对象的核心方法**

# compute_score
该函数是用来计算输入的一对或者多对句子对的得分并返回。


```python
import os
os.environ['CUDA_VISIBLE_DEVICES']='7'

from rag_retrieval import Reranker


ranker = Reranker('BAAI/bge-reranker-base',dtype='fp16',verbose=0)

pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]


scores = ranker.compute_score(pairs)

print(scores)
```

[-8.1484375, 6.18359375]

**compute_score函数的参数**
```python
    def compute_score(self, 
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: int = 256,
        max_length: int = 512,
        normalize: bool = False,
        enable_tqdm: bool = True,
    ):
```

**输入参数解释**
- sentence_pairs： 需要计算得分的一对或者多对句子对。
- batch_size： 模型一次前向推理的batch_size.在函数内部，会将sentence_pairs切分成多个batch_size来进行推理。
- max_length： 句子对的总长度，超过就会截断。
- normalize：是否会对计算出来的得分使用sigmod归一化到0-1之间。
- enable_tqdm：是否开启tqdm展示推理的进度。

对于LLM ranker 中的BAAI/bge-reranker-v2-minicpm-layerwise模型，可以在这里传递cutoff_layers指定推理的层数。其余模型不需要传递。
- cutoff_layers: list = None,
**返回解释**

如果输入是一对句子，那么返回一个float，代表这对句子的分值。如果输入是多对句子，那么返回一组fload的列表，是这一组的分值.



# rerank函数
该函数是用来计算query以及一组doc的得分，可以支持不同的长doc处理策略。

```python

import os
os.environ['CUDA_VISIBLE_DEVICES']='7'

from rag_retrieval import Reranker

ranker = Reranker('BAAI/bge-reranker-base',dtype='fp16',verbose=0)

query='what is panda?'

docs=['hi','The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']

doc_ranked = ranker.rerank(query,docs)
print(doc_ranked)

```
results=[Result(doc_id=1, text='The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.', score=6.18359375, rank=1), Result(doc_id=0, text='hi', score=-8.1484375, rank=2)] query='what is panda?' has_scores=True

**rerank的参数**

```python
    def rerank(self, 
        query: str, 
        docs: Union[List[str], str] = None,
        batch_size: int = 256,
        max_length: int = 512,
        normalize: bool = False,
        long_doc_process_strategy: str="max_score_slice",#['max_score_slice','max_length_truncation']
    ):  
```

**输入参数解释**
- query： query文本。
- docs： 一个或者一组doc的文本。
- batch_size： 模型一次前向推理的batch_size。
- max_length： 句子对的总长度，超过就会截断。
- normalize：是否会对计算出来的得分使用sigmod归一化到0-1之间。
- long_doc_process_strategy： 对于长doc处理的逻辑，可以选择
    - max_score_slice：将长doc按照长度切分，分别计算query和所有子doc的分数，取query与最大的子doc的分数作为query和整个doc的分数。
    - max_length_truncation：query加doc的长度超过max_length就会截断,来计算分数。

对于LLM ranker 中的BAAI/bge-reranker-v2-minicpm-layerwise模型，可以在这里传递cutoff_layers指定推理的层数。其余模型不需要传递。
- cutoff_layers: list = None,

**返回解释**

返回是一个RankedResults对象，其主要的属性有：results: List[Result]。一组Result对象，而Result的属性有：
- doc_id: Union[int, str]
- text: str
- score: Optional[float] = None
- rank: Optional[int] = None

RankedResults对象也有一些常见的方法如top_k:按照score返回top_k个Result.get_score_by_docid:输入doc在输入的顺序，得到对应的score。


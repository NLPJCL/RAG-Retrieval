# 利用LLM合成训练数据

## 1.查询-文档相似度分数合成

我们结合一个RAG的评估工具 [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) 进行实现和测试。 

### 1.1 索引建立
请先按照FlashRAG的文档安装好`flashrag`，以及建立 [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) 模型的 [wiki-18.jsonl](https://huggingface.co/datasets/ignore/FlashRAG_datasets/blob/main/retrieval-corpus/wiki-18.jsonl.gz) 文档的faiss索引。

### 1.2 数据集构建

修改 `flashrag_config.yaml` 中以下内容：
* `index_path` 改为上一步得到的索引文件的目录
* `llama2-7B-chat` 的路径改为你需要的LLM，理论上任意hf的LLM都可以适用
* `data_dir` 改为 [FlashRAG_datasets](https://huggingface.co/datasets/ignore/FlashRAG_datasets) 下载到本地的路径
* `corpus_path` 改为 wiki-18.jsonl 对应的路径

这里我们以NQ的测试集为例，得到基于LLaMA-3-Instruct的输出概率得到的数据集。这里的计算主要参考了 [REPLUG](https://arxiv.org/abs/2301.12652)

使用如下命令完成数据集的建立：
```shell
python3 get_lm_probs_dataset.py \ 
--dataset_name nq \
--split test \
--num 4000 \ # 查询数量
--gpu_id 0 \
--output lmsft.jsonl \ # jsonl 输出路径
--topk 20
```

得到的jsonl文件数据格式如下，query为用户查询，pos为若干文档，scores为每个文档相对于query的分数
```json
{"query":"xxx", "pos":["yyy" ,"zzz"], "scores": [0.2, 0.8]}
...
```

### 1.3 训练和评估

我们使用`train/train_embeddings.py`使用此数据进行训练，即可得到面向NQ任务的专属检索器

我们将FlashRAG中的检索器地址换成我们训练的检索器，并重新建立索引在NQ上测试，会得到如下结果，微调后的模型在NQ测试的表现明显优于原方法，在几个先进方法基础上都提升了5个点左右。

|                     Method                      | NQ EM Score | NQ F1 Score |
|:-----------------------------------------------:|:-----------:|:-----------:|
|                    Naive RAG                    |    36.09    |    47.23    |
|                 + **finetune**                  |  **41.50**  |  **52.69**  |
|   [REPLUG](https://arxiv.org/abs/2301.12652)    |    31.36    |    41.53    |
|                 + **finetune**                  |  **36.65**  |  **46.78**  |
| [Iter-Retgen](https://arxiv.org/abs/2305.15294) |    37.06    |    47.81    |
|                 + **finetune**                  |  **42.02**  |  **53.15**  |
|                    [SURE](https://arxiv.org/abs/2404.13081)                     |    37.62    |    49.24    |
|                 + **finetune**                  |  **41.25**  |  **53.20**  |

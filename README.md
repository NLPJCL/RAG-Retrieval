<h1 align="center">RAG-Retrieval</h1>
<p align="center">
    <a href="https://github.com/NLPJCL/RAG-Retrieval">
            <img alt="Build" src="https://img.shields.io/badge/Contribution-Welcome-blue">
    </a>
    <a href="https://github.com/NLPJCL/RAG-Retrieval/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
    </a>
</p>
<h4 align="center">
    <p>
        <a href=#创建环境>创建环境</a> |
        <a href="#微调模型">微调模型</a> |
        <a href=#实验结果>实验结果</a> |
        <a href="#license">License</a> 
    <p>
</h4>

RAG-Retrieval 提供了全链路的RAG检索微调代码，支持微调任意开源的RAG检索模型，包括向量（embedding、图a）、迟交互式模型（colbert、图d）、交互式模型（cross encoder、图c）。
![ColBERT](pictures/models.png)
# 最新更新

- 3/18/2024:发布RAG-Retrieval [RAG-Retrieval知乎介绍](https://zhuanlan.zhihu.com/p/683483778)

# 创建环境
    
```bash
conda create -n rag-retrieval python=3.8 && conda activate rag-retrieval
#为了避免自动安装的torch与本地的cuda不兼容，建议进行下一步之前先手动安装本地cuda版本兼容的torch。
pip install -r requirements.txt 
```


# 微调模型

##  向量（embedding）模型
- 支持微调任意开源的embedding模型（bge,m3e等等）

- 支持对两种数据进行微调：
    -  query和正例（负例采用batch内随机负例），
    -  query和正例以及难负例。（负例为对应的难负例，以及batch内随机负例）

微调embedding模型流程
```bash
cd ./rag-retrieval/embedding
bash train_embedding.sh
```

## 迟交互式（colbert）模型

- 支持微调开源的bge-m3e模型中的colbert。
- 支持query和正例以及难负例。（负例为对应的难负例，以及batch内随机负例）

微调colbert模型流程
```bash
cd ./rag-retrieval/colbert
bash train_colbert.sh
```
## 排序（reranker,cross encoder）模型
- 支持微调任意开源的reranker模型（例如，bge-rerank、bce-rerank等）
- 支持两种数据进行微调：
    - query和doc的相关性为二分类（1代表相关、0代表不相关）
    - query和doc的相关性为四分类。（3，2，1，0，相关性依次降低。）

微调reranker模型流程
```bash
cd ./rag-retrieval/reranker
bash train_reranker.sh
```


# 实验结果

说明：结果仅供参考，训练模型只是为了验证RAG-Retrieval的训练代码是否正确。相比的两个模型，都是很好的开源模型。而且他们都是多语言模型，而笔者训练的模型仅仅支持中文。

## reranker模型在 MTEB Reranking 任务的结果


|      **Model**       |  **Model Size(GB)**  |**T2Reranking** | **MMarcoReranking** | **CMedQAv1** | **CMedQAv2** | **Avg** |
|:-----------:|:----------:|:----------:|:-------------:|:--------------:|:---------------:| :---------------:|
|   bge-reranker-base   |  1.11 | 67.28    |      35.46     |      81.27      |       84.10      | 67.03
| bce-reranker-base_v1 |   1.11 |70.25    |      34.13     |      79.64      |       81.31      | 66.33
| rag-retrieval-reranker |  0.41 | 67.33    |      31.57     |      83.54     |       86.03     | 67.12

其中，rag-retrieval-reranker是我们使用RAG-Retrieval代码在hfl/chinese-roberta-wwm-ext模型上训练所得，训练数据使用bge-rerank模型的训练数据.

## colbert模型在 MTEB Reranking 任务的结果

|      **Model**  | **Model Size(GB)**  | **Dim**  | **T2Reranking** | **MMarcoReranking** | **CMedQAv1** | **CMedQAv2** | **Avg** |
|:-----------: |:----------:|:----------:|:----------:|:-------------:|:--------------:|:---------------:| :---------------:|
|   bge-m3-colbert   | 2.24 | 1024 | 66.82 | 26.71    |      75.88     |      76.83      |      61.56      
| rag-retrieval-colbert | 0.41 |  1024|  66.85    |      31.46     |      81.05     |       84.22     | 65.90

其中，rag-retrieval-colbert是我们使用RAG-Retrieval代码在hfl/chinese-roberta-wwm-ext模型上训练所得，训练数据使用bge-rerank模型的训练数据.


## 用领域内数据微调开源的BGE系列模型

|      **Model**  | **T2ranking**  | |
|:-----------: |:----------:|:----------:|
|   bge-v1.5-embedding   | 66.49|  | 
|   bge-v1.5-embedding **finetune**    | 67.15 | **+0.66** | 
|   bge-m3-colbert   | 66.82|  | 
|   bge-m3-colbert **finetune**    | 67.22 | **+0.40** | 
|   bge-reranker-base   | 67.28|  | 
|   bge-reranker-base  **finetune**    | 67.57 | **+0.29** | 

后面带有finetune的代表我们使用RAG-Retrieval在对应开源模型的基础上继续微调所得，训练数据使用T2-Reranking的训练集。

## License
RAG-Retrieval is licensed under the [MIT License](https://github.com/NLPJCL/RAG-Retrieval/blob/master/LICENSE). 

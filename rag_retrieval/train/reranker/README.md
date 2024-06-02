

# 安装环境
```bash
conda create -n rag-retrieval python=3.8 && conda activate rag-retrieval
#为了避免自动安装的torch与本地的cuda不兼容，建议进行下一步之前先手动安装本地cuda版本兼容的torch。
pip install -r requirements.txt 
```

# 微调模型
在安装好依赖后，我们通过具体的示例来展示如何利用我们自己的数据来微调开源的排序模型。(bge-rerank-base(v1),bce-rerank-basev1),或者你可以从BERT类模型(chinese-roberta-wwm-ext)开始从零训练自己的排序模型。

# 数据格式


对于排序模型，支持以下两种数据进行微调：

- 标注数据的相关性为二分类数据：query和doc的关系只有正例和负例。可以参考[example_data](https://github.com/NLPJCL/RAG-Retrieval/blob/master/example_data/t2rank_100.jsonl)的jsonl文件。
```
{"query": str, "pos": List[str], "neg":List[str]}
```
在训练中，我们采用二分类交叉熵(bce loss)来进行训练，会把queyr和正例组成pair，标签为1，query和负例组成pair，标签为0。因此在预测时，最终得到的score经过sigmoid后为0-1之间的值。
- 标注数据的相关性为四分类数据：query和doc的关系为3，2，1，0，
```
query\tdoc\tlabel
```
对于该种数据，在训练中，为了让模型学到不同档次的信息，例如3>2>1>0,因此，我们将任务建模成回归任务。在代码中，我们将相关性标签通过label2id做了如下的映射,label2id = {'3':1,'2':0.75,'1':0.5,'0':0.25}。因此，因此在预测时，最终得到的score后也为0-1之间的值。



# 训练
执行bash train_reranker.sh即可，下面是train_reranker.sh执行的代码。

```bash
 CUDA_VISIBLE_DEVICES="0"  nohup  accelerate launch --config_file ../../../config/default_fsdp.yaml train_reranker.py  \
--model_name_or_path "hfl/chinese-roberta-wwm-ext" \
--dataset "../../../example_data/t2rank_100.jsonl" \
--output_dir "./output/t2ranking_100_example" \
--loss_type "classfication" \
--batch_size 32 \
--lr 5e-5 \
--epochs 2 \
--num_labels 1 \
--log_with  'wandb' \
--save_on_epoch_end 1 \
--warmup_proportion 0.1 \
--gradient_accumulation_steps 1 \
--max_len 512 \
 >./logs/t2ranking_100_example.log &
```

**参数解释**
- model_name_or_path:开源的embedding模型的名称或下载下来的服务器位置.（可以是：BAAI/bge-reranker-base,maidalun1020/bce-reranker-base_v1，也可以从普通的bert类模型开始训练，例如hfl/chinese-roberta-wwm-ext）
- loss_type：可以在classfication以及regression选择。其中classfication是第一种数据格式，采用bce loss训练。regression是第二种数据格式，采用mse loss训练。
- save_on_epoch_end:是否每个epoch都将模型存储下来。
- log_with：可视化工具，如果不设置用默认参数,也不会报错。
- batch_size :每个batch query-doc的piar对的数量。
- lr :学习率，一般1e-5到5e-5之间。

默认使用fsdp来支持多卡训练模型，以下是配置文件的示例
- [default_fsdp](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/default_fsdp.yaml)。如果要在chinese-roberta-wwm-ext的基础上从零开始训练的排序，采用该配置文件。
- [xlmroberta_default_config](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/xlmroberta_default_config.yaml),如果要在bge-reranker-base和bce-reranker-base_v1的基础上进行微调，采用该配置文件，因为两者在多语言的xlmroberta的基础上训练而来。

多卡训练配置文件修改:
- 修改train_reranker.sh的CUDA_VISIBLE_DEVICES="0"为你想要设置的多卡。
- 修改上述提到的配置文件的num_processes为你想要跑的卡的数量。

# 加载模型进行预测

对于保存的模型，你可以很容易加载模型来进行预测。在model.py里，我们给了一个示例如何加载以及预测。


```python
ckpt_path='maidalun1020/bce-reranker-base_v1'
device = 'cuda:0'
cross_encode=CrossEncoder.from_pretrained(ckpt_path,num_labels=1,cuda_device=device)
cross_encode.eval()
cross_encode.model.to(device)

input_lst=[
    ['我喜欢中国','我喜欢中国'],
    ['我喜欢美国','我一点都不喜欢美国'],
    ['泰山要多长时间爬上去','爬上泰山需要1-8个小时，具体的时间需要看个人的身体素质。专业登山运动员可能只需要1个多小时就可以登顶，有些身体素质比较低的，爬的慢的就需要5个多小时了。']]

res=cross_encode.compute_score(input_lst)

print(torch.sigmoid(res[0]))
print(torch.sigmoid(res[1]))
print(torch.sigmoid(res[2]))

```


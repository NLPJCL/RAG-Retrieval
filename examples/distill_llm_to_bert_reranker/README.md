

#通过下面的命令，将train_data_path的query和pos和neg中的doc，两两组成pair，通过训练好的llm模型(ckpt_path)预测标签为1的score，再输出到distill_train_data_path中。
```bash

train_data_path="../../example_data/t2rank_100.jsonl"
ckpt_path=""
distill_train_data_path="../../example_data/t2rank_100.distill.jsonl"

python create_distill_data.py $train_data_path $ckpt_path $distill_train_data_path

```

- train_data_path的格式，query和doc的关系只有正例和负例。可以参考[example_data](https://github.com/NLPJCL/RAG-Retrieval/blob/master/example_data/t2rank_100.jsonl)的jsonl文件。
```
{"query": str, "pos": List[str], "neg":List[str]}
```

- distill_train_data_path的格式：可以参考[example_data](https://github.com/NLPJCL/RAG-Retrieval/blob/master/example_data/t2rank_100.distill.jsonl)的jsonl文件。
```
{"query": str, "content": str, "score":str(float)}
```




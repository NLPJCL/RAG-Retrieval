

#Use the following command to extract the embedding of the teacher model from the query, pos, and neg of train_data_path, and then output it to distill_train_data_path.
```bash

train_data_path="../../example_data/t2rank_100.jsonl"
ckpt_path="Alibaba-NLP/gte-Qwen2-7B-instruct"
distill_train_data_path="../../../example_data/t2rank_100.embedding"
nohup python   create_distill_data.py $train_data_path $ckpt_path $distill_train_data_path >./distill.log &

```

- train_data_path: refer to[example_data](https://github.com/NLPJCL/RAG-Retrieval/blob/master/example_data/t2rank_100.jsonl)的jsonl文件。
```
{"query": str, "pos": List[str], "neg":List[str]}
```
- t2rank_100.embedding.mmap:  np.memmap(teacher embedding)
- t2rank_100.embedding.text.jsonl: {"query}




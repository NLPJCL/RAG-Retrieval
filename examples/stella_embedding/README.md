
Refer to create_distill_data.sh to extract the embeddings of teacher1 and teacher2 in turn.

Tips:
- If the embedding model you want to distill requires instructions, please modify the create_distill_data.py to add them.
- For create_distill_data.py input file: train_data_path, No need to explain query, pos and neg are the doc texts to be distilled.
There are two options,  
  1. They can be doc text that is irrelevant to the query.
  2. They can be positive and hard negative examples of the query. If this is the case, and there is no shuffling during training, the distillation effect will be slightly better.

input:
- train_data_path: refer to[example_data](https://github.com/NLPJCL/RAG-Retrieval/blob/master/example_data/t2rank_100.jsonl).
```
{"query": str, "pos": List[str], "neg":List[str]}
```

output:
- t2rank_100.embedding.model_name.mmap: np.memmap(teacher embedding)
- t2rank_100.jsonl.text.jsonl: (If the embedding model you are training requires instructions for query, you can add the prompt_for_query field.)
```
{"query": str, "prompt_for_query"(optional): str}
```

For teachers who want to distill two teachers, they can be concat through concate_two_teacher_embedding.sh








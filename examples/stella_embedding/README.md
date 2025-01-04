

# Create distill data
Refer to create_distill_data.sh to extract the embeddings of teacher1 and teacher2 in turn.

**Special attention**:
- If the embedding model you want to distill requires instructions, please modify the create_distill_data.py to add them.

create_distill_data.sh
```bash
train_data_path="../../example_data/t2rank_100.jsonl"
ckpt_path="lier007/xiaobu-embedding-v2"
teacher1_distill_path="../../example_data/t2rank_100.embedding.xiaobu-embedding-v2.mmap"
save_text=1

CUDA_VISIBLE_DEVICES="7" nohup python create_distill_data.py $train_data_path $ckpt_path $teacher1_distill_path $save_text >./distill_xiaobu.log &
```

input:
- train_data_path: refer to [example_data](https://github.com/NLPJCL/RAG-Retrieval/blob/master/example_data/t2rank_100.jsonl).
```
{"query": str, "pos": List[str], "neg":List[str]}
```

- For train_data_path, pos and neg are the doc texts to be distilled.
There are two options,
  1. They can be doc text that is irrelevant to the query.
  2. They can be positive and hard negative examples of the query. If this is the case, and there is no shuffling during training, the distillation effect will be slightly better.

output:
- teacher1_distill_path(embedding): np.memmap(teacher embedding)
- teacher1_distill_text: if save_text==1,it will be in train_data_path+.text.jsonl.
  - the embedding model you are training requires instructions for query, you can add the prompt_for_query field.
```
{"query": str, "prompt_for_query"(optional): str}
```


# concate_two_teacher_embedding
If you want to distill two embedding models, please refer to concate_two_teacher_embedding.sh to concat them.

```bash
train_data_nums=2087
teacher1_dims=1792
teacher2_dims=1792
teacher1_distill_path="../../example_data/t2rank_100.embedding.xiaobu-embedding-v2.mmap"
teacher2_distill_path="../../example_data/t2rank_100.embedding.conan-embedding-v1.mmap"
two_teacher_distill_path="../../example_data/t2rank_100.embedding.conan.xiaobu.mmap"


python concate_two_teacher_embedding.py \
    $teacher1_distill_path \
    $teacher2_distill_path \
    $train_data_nums \
    $teacher1_dims \
    $teacher2_dims \
    $two_teacher_distill_path
```








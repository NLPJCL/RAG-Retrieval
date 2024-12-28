train_data_path="../../example_data/t2rank_100.jsonl"
ckpt_path="Alibaba-NLP/gte-Qwen2-7B-instruct"
distill_train_data_path="../../../example_data/t2rank_100.embedding"

nohup python   create_distill_data.py $train_data_path $ckpt_path $distill_train_data_path >./distill.log &
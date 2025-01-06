#!/bin/bash

if [ ! -d "./output" ]; then
    mkdir -p ./output
    echo "mkdir output"
fi

if [ ! -d "./logs" ]; then
    mkdir -p ./logs
    echo "mkdir logs"
fi

#bert-based embedding 
 CUDA_VISIBLE_DEVICES="0,1"   nohup  accelerate launch \
 --config_file ../../../config/default_fsdp.yaml \
 train_embedding.py  \
 --config ./config/training_embedding.yaml  \
 >./logs/t2ranking_100_example_bert.log &


#llm-based embedding
#  CUDA_VISIBLE_DEVICES="0,1"   nohup  accelerate launch \
#  --config_file ../../../config/deepspeed/deepspeed_zero2.yaml \
#  train_embedding.py  \
#  --config ./config/training_embedding.yaml  \
#  >./logs/t2ranking_100_example_llm.log &

#distill (stella method)
#  CUDA_VISIBLE_DEVICES="0,1"   nohup  accelerate launch \
#  --config_file ../../../config/default_fsdp.yaml \
#  train_embedding.py  \
#  --config ./config/distill_embedding.yaml  \
#  >./logs/t2ranking_100_example_bert_distill.log &


#bert-based embedding (without mrl)
# CUDA_VISIBLE_DEVICES="0,1"   nohup  accelerate launch --config_file ../../../config/default_fsdp.yaml train_embedding.py  \
# --model_name_or_path "BAAI/bge-base-zh-v1.5" \
# --dataset "../../../example_data/t2rank_100.jsonl" \
# --output_dir "./output/t2ranking_100_example" \
# --batch_size 4 \
# --lr 2e-5 \
# --epochs 2 \
# --save_on_epoch_end 1 \
# --gradient_accumulation_steps 24  \
# --log_with 'wandb' \
# --warmup_proportion 0.1 \
# --neg_nums 15 \
# --temperature 0.02 \
# --query_max_len 128 \
# --passage_max_len 512 \
#  >./logs/t2ranking_100_example.log &

#bert-based embedding (with mrl)
#  CUDA_VISIBLE_DEVICES="0,1"   nohup  accelerate launch --config_file ../../../config/default_fsdp.yaml train_embedding.py  \
# --model_name_or_path "BAAI/bge-base-zh-v1.5" \
# --dataset "../../../example_data/t2rank_100.jsonl" \
# --output_dir "./output/t2ranking_100_example_mrl1792" \
# --batch_size 4 \
# --lr 2e-5 \
# --epochs 2 \
# --use_mrl \
# --mrl_dims "128, 256, 512, 768, 1024, 1280, 1536, 1792" \
# --save_on_epoch_end 1 \
# --gradient_accumulation_steps 24  \
# --log_with 'wandb' \
# --warmup_proportion 0.1 \
# --neg_nums 15 \
# --temperature 0.02 \
# --query_max_len 128 \
# --passage_max_len 512 \
#  >./logs/t2ranking_100_example_mrl1792.log &

#llm-based embedding
# CUDA_VISIBLE_DEVICES="0,1"   nohup  accelerate launch --config_file ../../../config/deepspeed/deepspeed_zero2.yaml train_embedding.py  \
# --model_name_or_path "Alibaba-NLP/gte-Qwen2-7B-instruct" \
# --dataset "../../../example_data/t2rank_100.jsonl" \
# --output_dir "./output/t2ranking_100_example" \
# --batch_size 4 \
# --lr 2e-5 \
# --epochs 2 \
# --save_on_epoch_end 1 \
# --gradient_accumulation_steps 24  \
# --log_with 'wandb' \
# --warmup_proportion 0.05 \
# --neg_nums 15 \
# --temperature 0.02 \
# --query_max_len 256 \
# --passage_max_len 1024 \
#  >./logs/t2ranking_100_example.log &

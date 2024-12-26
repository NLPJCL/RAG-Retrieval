#!/bin/bash

if [ ! -d "./output" ]; then
    mkdir -p ./output
    echo "mkdir output"
fi

if [ ! -d "./logs" ]; then
    mkdir -p ./logs
    echo "mkdir logs"
fi

CUDA_VISIBLE_DEVICES="0,1"   nohup  accelerate launch --config_file ../../../config/default_fsdp.yaml train_colbert.py  \
--model_name_or_path "hfl/chinese-roberta-wwm-ext" \
--dataset "../../../example_data/t2rank_100.jsonl" \
--output_dir "./output/t2ranking_100_example" \
--batch_size 4 \
--lr 5e-6 \
--epochs 2 \
--save_on_epoch_end 1 \
--gradient_accumulation_steps 12 \
--temperature 0.02 \
--query_max_len 128 \
--passage_max_len 512 \
--colbert_dim 768 \
--neg_nums 15 \
--log_with 'wandb' \
--warmup_proportion 0.1 \
 >./logs/t2ranking_100_example.log &

#!/bin/bash

if [ ! -d "./output" ]; then
    mkdir -p ./output
    echo "mkdir output"
fi

if [ ! -d "./logs" ]; then
    mkdir -p ./logs
    echo "mkdir logs"
fi

# bert model,fsdp(ddp) loss_type "point_ce"
CUDA_VISIBLE_DEVICES="0"  nohup  accelerate launch --config_file ../../../config/default_fsdp.yaml train_reranker.py  \
--model_name_or_path "hfl/chinese-roberta-wwm-ext" \
--train_dataset "../../../example_data/t2rank_100.jsonl" \
--val_dataset "../../../example_data/t2rank_100.jsonl" \
--output_dir "./output/t2ranking_100_example" \
--model_type "SeqClassificationRanker" \
--loss_type "point_ce" \
--batch_size 32 \
--lr 5e-5 \
--epochs 2 \
--num_labels 1 \
--log_with  'wandb' \
--save_on_epoch_end 1 \
--warmup_proportion 0.1 \
--gradient_accumulation_steps 3 \
--max_len 512 \
--gradient_accumulation_steps 4 \
--mixed_precision "bf16" \
 >./logs/t2ranking_100_example.log &


# llm model, deepspeed(zero1-2, not for zero3) loss_type "point_ce"
 CUDA_VISIBLE_DEVICES="4,5,6,7"  nohup  accelerate launch --config_file ../../../config/deepspeed/deepspeed_zero2.yaml train_reranker.py  \
--model_name_or_path "Qwen/Qwen2.5-1.5B" \
--train_dataset "../../../example_data/t2rank_100.jsonl" \
--val_dataset "../../../example_data/t2rank_100.jsonl" \
--output_dir "./output/t2ranking_100_example_llm_decoder" \
--model_type "SeqClassificationRanker" \
--loss_type "point_ce" \
--batch_size 8 \
--lr 5e-5 \
--epochs 2 \
--num_labels 1 \
--log_with 'wandb' \
--save_on_epoch_end 1 \
--warmup_proportion 0.1 \
--gradient_accumulation_steps 3 \
--max_len 512 \
--gradient_accumulation_steps 8 \
--mixed_precision "bf16" \
 >./logs/t2ranking_100_example_llm_decoder.log &


# bert model, fsdp(ddp), distill(distill_llama_to_bert) loss_type "point_mse" or "point_ce"
 CUDA_VISIBLE_DEVICES="0"  nohup  accelerate launch --config_file ../../../config/default_fsdp.yaml train_reranker.py  \
--model_name_or_path "hfl/chinese-roberta-wwm-ext" \
--train_dataset "../../../example_data/t2rank_100_distill.jsonl" \
--val_dataset "../../../example_data/t2rank_100_distill.jsonl" \
--output_dir "./output/t2ranking_100_example_distill" \
--model_type "SeqClassificationRanker" \
--loss_type "point_mse" \
--batch_size 32 \
--lr 5e-5 \
--epochs 2 \
--num_labels 1 \
--log_with  'wandb' \
--save_on_epoch_end 1 \
--warmup_proportion 0.1 \
--max_len 512 \
--gradient_accumulation_steps 3 \
--mixed_precision "fp16" \
 >./logs/t2ranking_100_example_distill.log &

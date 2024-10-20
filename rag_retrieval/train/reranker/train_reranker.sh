#!/bin/bash

if [ ! -d "./output" ]; then
    mkdir -p ./output
    echo "mkdir output"
fi

if [ ! -d "./logs" ]; then
    mkdir -p ./logs
    echo "mkdir logs"
fi

#model_bert,fsdp(ddp)
 CUDA_VISIBLE_DEVICES="0"  nohup  accelerate launch --config_file ../../../config/default_fsdp.yaml train_reranker.py  \
--model_name_or_path "hfl/chinese-roberta-wwm-ext" \
--dataset "../../../example_data/t2rank_100.jsonl" \
--output_dir "./output/t2ranking_100_example" \
--model_type "cross_encoder" \
--loss_type "classfication" \
--batch_size 32 \
--lr 5e-5 \
--epochs 2 \
--num_labels 1 \
--log_with  'wandb' \
--save_on_epoch_end 1 \
--warmup_proportion 0.1 \
--gradient_accumulation_steps 3 \
--max_len 512 \
 >./logs/t2ranking_100_example.log &


#model_llm,deepspeed(zero1-3)
 CUDA_VISIBLE_DEVICES="4,5,6,7"  nohup  accelerate launch --config_file ../../../config/deepspeed/deepspeed_zero2.yaml train_reranker.py  \
--model_name_or_path "Qwen/Qwen2.5-1.5B" \
--dataset "../../../example_data/t2rank_100.jsonl" \
--output_dir "./output/t2ranking_100_example_llm_decoder" \
--model_type "llm_decoder" \
--loss_type "classfication" \
--mixed_precision 'bf16' \
--batch_size 32 \
--lr 5e-5 \
--epochs 2 \
--num_labels 1 \
--log_with  'wandb' \
--save_on_epoch_end 1 \
--warmup_proportion 0.1 \
--gradient_accumulation_steps 3 \
--max_len 512 \
 >./logs/t2ranking_100_example_llm_decoder.log &


#model_bert,fsdp(ddp),distill(distill_llama_to_bert)
 CUDA_VISIBLE_DEVICES="0"  nohup  accelerate launch --config_file ../../../config/default_fsdp.yaml train_reranker.py  \
--model_name_or_path "hfl/chinese-roberta-wwm-ext" \
--dataset "../../../example_data/t2rank_100.distill.jsonl" \
--output_dir "./output/t2ranking_100_example_distill" \
--model_type "cross_encoder" \
--loss_type "regression_mse" \
--batch_size 32 \
--lr 5e-5 \
--epochs 2 \
--num_labels 1 \
--log_with  'wandb' \
--save_on_epoch_end 1 \
--warmup_proportion 0.1 \
--gradient_accumulation_steps 3 \
--max_len 512 \
 >./logs/t2ranking_100_example_distill.log &

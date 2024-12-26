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
CUDA_VISIBLE_DEVICES="0,1" nohup accelerate launch \
--config_file ../../../config/xlmroberta_default_config.yaml \
train_reranker.py \
--config config/training_bert.yaml \
>./logs/training_bert.log &


# llm model, deepspeed(zero1-2, not for zero3) loss_type "point_ce"
CUDA_VISIBLE_DEVICES="0,1" nohup accelerate launch \
--config_file ../../../config/deepspeed/deepspeed_zero1.yaml \
train_reranker.py \
--config config/training_llm.yaml \
>./logs/training_llm_deepspeed1.log &


# bert model, fsdp(ddp), distill(distill_llama_to_bert) loss_type "point_mse" or "point_ce"
CUDA_VISIBLE_DEVICES="0,1" nohup accelerate launch \
--config_file ../../../config/xlmroberta_default_config.yaml \
train_reranker.py \
--config config/distilling_bert.yaml \
>./logs/distilling_bert.log &



if [ ! -d "./output" ]; then
    mkdir -p ./output
    echo "mkdir output"
fi

if [ ! -d "./logs" ]; then
    mkdir -p ./logs
    echo "mkdir logs"
fi

 CUDA_VISIBLE_DEVICES="0"  nohup  accelerate launch --config_file ../../config/default_fsdp.yaml train_reranker.py  \
--model_name_or_path "hfl/chinese-roberta-wwm-ext" \
--dataset "../../example_data/t2rank_100.json" \
--output_dir "./output/t2ranking_100_example" \
--loss_type "classfication" \
--batch_size 32 \
--lr 5e-5 \
--epochs 2 \
--num_labels 1 \
--log_with  'wandb' \
--save_on_epoch_end 1 \
--warmup_proportion 0.1 \
--gradient_accumulation_steps 1 \
--max_len 512 \
 >./logs/t2ranking_100_example.log &

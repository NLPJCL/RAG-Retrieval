
#distill teacher1
train_data_path="../../example_data/t2rank_100.jsonl"
ckpt_path="lier007/xiaobu-embedding-v2"
teacher1_distill_path="../../example_data/t2rank_100.embedding.xiaobu-embedding-v2.mmap"
save_text=1

CUDA_VISIBLE_DEVICES="7" nohup python create_distill_data.py $train_data_path $ckpt_path $teacher1_distill_path $save_text >./distill_xiaobu.log &


#distill teacher2
train_data_path="../../example_data/t2rank_100.jsonl"
ckpt_path="TencentBAC/Conan-embedding-v1"
teacher2_distill_path="../../example_data/t2rank_100.embedding.conan-embedding-v1.mmap"
save_text=0

CUDA_VISIBLE_DEVICES="7" nohup python create_distill_data.py $train_data_path $ckpt_path $teacher2_distill_path $save_text >./distill_Conan-embedding-v1.log &


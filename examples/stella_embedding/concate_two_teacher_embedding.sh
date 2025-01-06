#concat teacher1 and teacher2
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
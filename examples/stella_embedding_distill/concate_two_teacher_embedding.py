import json
import sys
from tqdm.autonotebook import trange
import torch
import numpy as np

from sentence_transformers import SentenceTransformer

if __name__=="__main__":

    if len(sys.argv) != 7:
        print(
            "Parameter error"
        )
        sys.exit(0)

    teacher1_distill_train_data_path = sys.argv[1]
    teacher2_distill_train_data_path = sys.argv[2]
    train_data_nums = sys.argv[3]
    teacher1_dims = sys.argv[4]
    teacher2_dims = sys.argv[5]
    two_teacher_distill_train_data_path = sys.argv[6]

    train_data_nums = int(train_data_nums)
    teacher1_dims= int(teacher1_dims)
    teacher2_dims = int(teacher2_dims)

    teacher1_mmap_array = np.memmap(teacher1_distill_train_data_path, dtype='float32', mode='r', shape=(train_data_nums, teacher1_dims))
    teacher2_mmap_array = np.memmap(teacher2_distill_train_data_path, dtype='float32', mode='r', shape=(train_data_nums, teacher2_dims))
    
    two_teacher_mmap_array = np.memmap(two_teacher_distill_train_data_path, dtype='float32', mode='w+', shape=(train_data_nums,teacher1_dims+teacher2_dims))

    batch_size=100000
    for start_index in trange(0, train_data_nums, batch_size, desc="concat teacher1 and teacher2 embedding ..."):
        batch_teacher1_embedding = teacher1_mmap_array[start_index : start_index + batch_size]    
        batch_teacher2_embedding = teacher2_mmap_array[start_index : start_index + batch_size]
        two_teacher_mmap_array[start_index : start_index + batch_size] = np.concatenate((batch_teacher1_embedding,batch_teacher2_embedding), axis=-1)

    two_teacher_mmap_array.flush() 
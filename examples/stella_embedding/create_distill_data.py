


import tqdm
import json
from transformers import AutoTokenizer
import sys
from tqdm.autonotebook import trange
import torch
import numpy as np

from sentence_transformers import SentenceTransformer

def get_train_data(train_data_path):
    train_data = []
    train_data_text = []
    dic = set()
    with open(train_data_path) as f:
        for line in tqdm.tqdm(f):
            data_dic=json.loads(line.strip())
            query = data_dic['query']

            #Please note whether you need to add instructions to the distilled embedding model?
            if query not in dic:
                train_data.append(query)
                train_data_text.append(data_dic['query'])
                dic.add(query)
            if 'pos' in data_dic:
                for text_pos in data_dic['pos']:
                    if text_pos not in dic :
                        train_data.append(text_pos)
                        train_data_text.append(text_pos)
                        dic.add(text_pos)

            if 'neg' in data_dic:
                for text_neg in data_dic['neg']:
                    if text_neg not in dic :
                        train_data.append(text_neg)
                        train_data_text.append(text_neg)
                        dic.add(text_neg)
    print(len(train_data))
    print(len(dic))
    return train_data,train_data_text


if __name__=="__main__":

    if len(sys.argv) != 5:
        print(
            "Usage: python create_distill_data.py [train_data_path] [ckpt_path] [distill_train_data_path] [save_text]"
        )
        sys.exit(0)

    input_train_data_path = sys.argv[1]
    ckpt_path = sys.argv[2]
    distill_train_data_path = sys.argv[3]
    save_text = sys.argv[4]


    train_data, train_data_text = get_train_data(input_train_data_path)
    
    embedding_model = SentenceTransformer(
            ckpt_path,
            trust_remote_code=True,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
            }
        )

    multi_process_pool = embedding_model.start_multi_process_pool()

    embedding_model.max_seq_length = 512

    train_data_embedding_dim = embedding_model.get_sentence_embedding_dimension()

    mmap_array = np.memmap(distill_train_data_path, dtype='float32', mode='w+', shape=(len(train_data),train_data_embedding_dim))

    if save_text:
        f_w = open(input_train_data_path+'.text.jsonl','w')

    batch_size=100000
    for start_index in trange(0, len(train_data), batch_size, desc=""):
        batch_train_data = train_data[start_index : start_index + batch_size]

        train_data_embedding = embedding_model.encode_multi_process(batch_train_data,
            batch_size=256,
            pool=multi_process_pool,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        mmap_array[start_index : start_index + batch_size] = train_data_embedding
        mmap_array.flush()

        if save_text:
            batch_train_data_text = train_data_text[start_index : start_index + batch_size]
            for query in batch_train_data_text:
                dic = {}
                dic['query'] = query
                f_w.write(json.dumps(dic,ensure_ascii=False)+'\n')
        f_w.flush()
    embedding_model.stop_multi_process_pool(multi_process_pool)
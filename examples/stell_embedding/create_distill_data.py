


import tqdm
import json
from transformers import AutoTokenizer
import sys
from tqdm.autonotebook import trange
import torch
import numpy as np



from sentence_transformers import SentenceTransformer


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


# Each query must come with a one-sentence instruction that describes the task
task = 'Given a web search query, retrieve relevant passages that answer the query'

def get_train_data(train_data_path):
    train_data = []
    train_data_text = []
    dic = set()
    with open(train_data_path) as f:
        for line in tqdm.tqdm(f):
            data_dic=json.loads(line.strip())
            
            detiled_instruct = get_detailed_instruct(task,data_dic['query'])
            # print(detiled_instruct)
            if detiled_instruct not in dic:
                train_data.append(detiled_instruct)
                train_data_text.append(data_dic['query'])
                dic.add(detiled_instruct)


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

    if len(sys.argv) != 4:
        print(
            "Usage: python create_distill_data.py [train_data_path] [ckpt_path] [distill_train_data_path]"
        )
        sys.exit(0)

    input_train_data_path = sys.argv[1]
    ckpt_path = sys.argv[2]
    distill_train_data_path = sys.argv[3]

    train_data, train_data_text = get_train_data(input_train_data_path)
    
    #https://github.com/UKPLab/sentence-transformers/blob/52162f3f0b984a2032667a2332bf95f6aabcf033/sentence_transformers/SentenceTransformer.py#L444C14-L445C1
    embedding_model = SentenceTransformer(
            ckpt_path,
            trust_remote_code=True,
            device="cuda:7",
            model_kwargs={
                "torch_dtype": torch.bfloat16,  # fp16 容易计算出nan
            }
        )

    embedding_model.max_seq_length = 1024


    train_data_embedding = embedding_model.encode(train_data,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
        )
    print(train_data_embedding.shape)

    train_data_embedding = train_data_embedding.astype(dtype=np.float32)
    print(train_data[0])
    print(train_data_embedding[0])
    mmap_array = np.memmap(distill_train_data_path+'.mmap', dtype='float32', mode='w+', shape=(len(train_data),len(train_data_embedding[0])))

    batch_size=10000
    for start_index in trange(0, len(train_data), batch_size, desc="saving mmap_array..."):
        train_data_embedding_batch = train_data_embedding[start_index : start_index + batch_size]
        mmap_array[start_index : start_index + batch_size] = train_data_embedding_batch

    
    with open(distill_train_data_path+'.text.jsonl','w') as f_w:

        for idx, text in enumerate(train_data_text):
            dic = {}
            dic['query'] = text
            f_w.write(json.dumps(dic,ensure_ascii=False)+'\n')    
    mmap_array.flush()
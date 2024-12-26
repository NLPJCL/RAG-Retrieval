


import tqdm
import json
from transformers import AutoTokenizer
import sys
from model_llm_generate import LLMGenerateDecoder


def get_train_data(train_data_path):
    train_data = []
    with open(train_data_path) as f:
        for line in tqdm.tqdm(f):
            data_dic=json.loads(line.strip())

            if 'pos' in data_dic:
                for text_pos in data_dic['pos']:
                    train_data.append([data_dic['query'],text_pos])

            if 'neg' in data_dic:
                for text_neg in data_dic['neg']:
                    train_data.append([data_dic['query'],text_neg])

    print(len(train_data))
    return train_data

if __name__=="__main__":

    if len(sys.argv) != 4:
        print(
            "Usage: python create_distill_data.py [train_data_path] [ckpt_path] [distill_train_data_path]"
        )
        sys.exit(0)

    input_train_data_path = sys.argv[1]
    ckpt_path = sys.argv[2]
    distill_train_data_path = sys.argv[3]


    train_data = get_train_data(input_train_data_path)
    
    #  trained llm reranker
    llm_reranker = LLMGenerateDecoder.from_pretrained(ckpt_path)
    llm_reranker.eval()
    res_score = llm_reranker.compute_score(train_data)

    with open(distill_train_data_path,'w') as f_w:
        for idx, data in tqdm.tqdm(enumerate(train_data)):
            query = data[0]
            passage = data[1]
            dic = {}
            dic['query'] = query
            dic['content'] = passage
            dic['score'] = res_score[idx]
            f_w.write(json.dumps(dic,ensure_ascii=False)+'\n')

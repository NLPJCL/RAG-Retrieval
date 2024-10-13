

import os
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import tqdm
import json


class RankerDataset(Dataset):
    def __init__(self,
            train_data_path,
            tokenizer,
            negatives_nums = 15,
            max_len = 512,
        ):

        self.max_len = max_len
        self.negatives_nums=negatives_nums
        self.train_data=self.read_train_data(train_data_path)
        self.tokenizer = tokenizer
        


    def read_train_data(self,train_data_path):
        train_data = []
        with open(train_data_path) as f:
            for line in tqdm.tqdm(f):
                data_dic=json.loads(line.strip())

                if 'pos' in data_dic and 'neg' in data_dic:
                    
                    for text_pos in data_dic['pos']:
                        train_data.append([data_dic['query'],text_pos,1])
                    for text_neg in data_dic['neg'][:self.negatives_nums]:
                        train_data.append([data_dic['query'],text_neg,0])
        return train_data
    
    
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self,idx):
        return self.train_data[idx]

    def collate_fn(self,batch):

        all_batch_pairs=[]
        all_labels=[]
        for item in batch:
            all_batch_pairs.append([item[0],item[1]])
            all_labels.append(item[2])

        tokens = self.tokenizer.batch_encode_plus(all_batch_pairs,add_special_tokens=True,padding='max_length',truncation=True,
                            max_length=self.max_len,return_tensors='pt')

        label_batch = torch.tensor(all_labels,dtype=torch.float16)
                
        return tokens, label_batch


class RankerMulabelDataset(Dataset):
    def __init__(self,
            train_data_path,
            tokenizer,
            max_len=512,
        ):

        self.label2id = {'3':1,'2':0.75,'1':0.5,'0':0.25}
        self.max_len = max_len
        self.train_data = self.read_train_data(train_data_path)
        self.tokenizer = tokenizer


    def read_train_data(self,train_data_path):
        train_data = []

        with open(train_data_path) as f:
            for line in tqdm.tqdm(f):
                data_lst=line.strip().split('\t')
                train_data.append([data_lst[0],data_lst[1],self.label2id[data_lst[2]]])

        return train_data


    def __len__(self):
        return len(self.train_data)

    def __getitem__(self,idx):
        return self.train_data[idx]

    def collate_fn(self,batch):

        all_batch_pairs=[]
        all_labels=[]
        for item in batch:
            all_batch_pairs.append([item[0],item[1]])
            all_labels.append(item[2])

        tokens = self.tokenizer.batch_encode_plus(all_batch_pairs,add_special_tokens=True,padding='max_length',truncation=True,
                            max_length=self.max_len,return_tensors='pt')

        label_batch = torch.tensor(all_labels,dtype=torch.float16)
        
        return tokens,label_batch

class LLMRankerDataset(Dataset):
    def __init__(self,
            train_data_path,
            tokenizer,
            negatives_nums = 15,
            max_len = 512,
        ):
        self.max_len = max_len
        self.negatives_nums=negatives_nums
        self.train_data=self.read_train_data(train_data_path)
        self.tokenizer = tokenizer        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # make sure tokenizer is right pad in our logic
        self.tokenizer.padding_side = "right"

    def read_train_data(self,train_data_path):
        train_data = []
        count=0
        with open(train_data_path) as f:
            for line in tqdm.tqdm(f):
                data_dic=json.loads(line.strip())
                if 'pos' in data_dic:
                    for text_pos in data_dic['pos']:
                        train_data.append([data_dic['query'],text_pos,1])
                if 'neg' in data_dic:
                    for text_neg in data_dic['neg'][:self.negatives_nums]:
                        train_data.append([data_dic['query'],text_neg,0])
        return train_data
    
    
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self,idx):
        return self.train_data[idx]

    def collate_fn(self,batch):

        all_batch_pairs=[]
        all_labels=[]
        #Add a separator:\n between query and doc
        sep = '\n'
        for item in batch:
            all_batch_pairs.append([item[0]+sep,item[1]])
            all_labels.append(item[2])
        tokens = self.tokenizer.batch_encode_plus(all_batch_pairs,add_special_tokens=True,padding='max_length',truncation=True,
                            max_length=self.max_len,return_tensors='pt')
        label_batch = torch.tensor(all_labels,dtype=torch.float16)
                
        return tokens, label_batch



def test_RankerDataset():
    train_data_path='../../../example_data/t2rank_100.jsonl'

    model_name_or_path='Qwen/Qwen2.5-1.5B'

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    dataset = LLMRankerDataset(train_data_path,tokenizer)

    dataloader = DataLoader(dataset, batch_size=32, collate_fn=dataset.collate_fn)

    print(len(dataloader))

    for batch in tqdm.tqdm(dataloader):

        print(batch)
        print(tokenizer.batch_decode(batch[0]['input_ids'])[0])
        break


if __name__ == "__main__":

    test_RankerDataset()
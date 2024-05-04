import torch
import torch.nn as nn 
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
)
import os
from tqdm import tqdm 


class ColBERT(nn.Module):
    def __init__(self,
        hf_model = None,
        linear = None,
        tokenizer = None,
        cuda_device = 'cpu',
        mask_punctuation = True,
        temperature= 0.02,
    ):
        super().__init__()

        self.model = hf_model
        self.linear = linear
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.mask_punctuation = mask_punctuation
        
        self.cuda_device=cuda_device

        mask_symbol_list = [self.tokenizer.pad_token_id,self.tokenizer.cls_token_id]
        # if self.mask_punctuation:
        #     mask_symbol_list += [self.tokenizer.encode(symbol,add_special_tokens=False)[0] for symbol in string.punctuation]
        self.register_buffer('mask_buffer', torch.tensor(mask_symbol_list)) 
    
    def get_embedding(self,input_ids,attention_mask):

        embedding = self.model(
            input_ids,attention_mask,
        ).last_hidden_state
        embedding = self.linear(embedding[:,1:])

        puntuation_padding_mask = self.punctuation_padding_mask(input_ids)

        #把pad和cls部分的token的向量设置为0，不参与后续分数的计算。
        embedding = embedding * puntuation_padding_mask[:,1:].unsqueeze(-1)
        embedding = F.normalize(embedding,p=2,dim=-1)
                
        return embedding    


    def punctuation_padding_mask(self,input_ids):
        #[batch_size,seq_len]
        mask = (input_ids.unsqueeze(-1) == self.mask_buffer).any(dim=-1)
        mask = (~mask).float()
        return mask

    def score(self,query_embedding,doc_embedding,query_attention_mask):
        token_score = query_embedding @ doc_embedding.transpose(-1,-2)
        score = token_score.max(-1).values.sum(-1)
        if len(score.size())==1:
            score = score / query_attention_mask[:,1:].sum(-1)
        else:
            score = score / query_attention_mask[:,1:].sum(-1,keepdim=True)
        return score



    def forward(
        self,
        query_input_ids, # [batch_size,seq_len]
        query_attention_mask, # [batch_size,seq_len]
        pos_doc_input_ids, # [batch_size,seq_len]
        pos_doc_attention_mask, # [batch_size,seq_len]
        neg_doc_input_ids = None, # [batch_size*neg_nums,seq_len]
        neg_doc_attention_mask = None, # [batch_size*neg_nums,seq_len]
    ):  
        #[batch_size,seq_len,dim]
        query_embedding = self.get_embedding(query_input_ids,query_attention_mask)
        pos_doc_embedding = self.get_embedding(pos_doc_input_ids,pos_doc_attention_mask)

        #[batch_size]
        pos_score = self.score(query_embedding,pos_doc_embedding,query_attention_mask)

        res_dict = {}
        res_dict['score']=pos_score

        if neg_doc_input_ids is not None:

            neg_doc_embedding = self.get_embedding(neg_doc_input_ids,neg_doc_attention_mask) 
            #[batch_size,batch_size*neg_nums]
            #[batch_size,1,seq_len,dim],[1,batch_size*neg_nums,seq_len,dim]
            neg_score = self.score(query_embedding.unsqueeze(1),neg_doc_embedding.unsqueeze(0),query_attention_mask)

            loss_fct = nn.CrossEntropyLoss()
            #[batch_size]
            labels = torch.zeros(pos_score.shape[0], dtype=torch.long, device=pos_score.device) 
            #[batch_size,1+batch_size*neg_nums]
            pos_neg_score = torch.cat([pos_score.unsqueeze(1), neg_score], dim=1)/self.temperature
            loss = loss_fct(pos_neg_score,labels)
            res_dict['loss'] = loss
        return res_dict     

    @torch.no_grad()
    def compute_score(self,
        sentences_pairs,
        batch_size = 512,
        query_max_len = 128,
        passage_max_len = 512,
    ):
        '''
            sentences_pairs=[[query,title],[query1,title1],...]
        '''

        all_pred = []

        all_logits=[]
        for start_index in tqdm(range(0, len(sentences_pairs), batch_size), desc="Compute Scores"):

            sentences_batch=sentences_pairs[start_index:start_index+batch_size]
            query_input_ids,query_attention_mask,doc_input_ids,doc_attention_mask= self.preprocess(sentences_batch,query_max_len,passage_max_len)
            output=self.forward(query_input_ids,query_attention_mask,doc_input_ids,doc_attention_mask)
            logits = output['score']
            all_logits.extend(logits.cpu().numpy().tolist())

        return all_logits

    def preprocess(self,
        sentences,
        query_max_len = 128,
        passage_max_len = 512
    ):
    
        all_querys = []
        all_docs = []
        for item in sentences:
            all_querys.append(item[0])
            all_docs.append(item[1])

        all_query_tokens = self.tokenizer(all_querys,padding='max_length',truncation=True,
                            max_length=query_max_len,return_tensors='pt')

        all_doc_tokens = self.tokenizer(all_docs,padding='max_length',truncation=True,
                            max_length=passage_max_len,return_tensors='pt')

        query_input_ids = all_query_tokens['input_ids'].to(self.cuda_device)
        query_attention_mask = all_query_tokens['attention_mask'].to(self.cuda_device)

        doc_input_ids = all_doc_tokens['input_ids'].to(self.cuda_device)
        doc_attention_mask = all_doc_tokens['attention_mask'].to(self.cuda_device)
        return query_input_ids,query_attention_mask,doc_input_ids,doc_attention_mask


    def save_pretrained(self,
        save_dir
    ):
        def _trans_state_dict(state_dict):
            state_dict = type(state_dict)(
                {k: v.clone().cpu()
                    for k,
                    v in state_dict.items()})
            return state_dict
        self.model.save_pretrained(save_dir,state_dict=_trans_state_dict(self.model.state_dict()),safe_serialization=False)        
        torch.save(_trans_state_dict(self.linear.state_dict()), os.path.join(save_dir, 'colbert_linear.pt'))


    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        colbert_dim,
        cuda_device = 'cpu',
        mask_punctuation = True,
        temperature = 0.02
    ):  
        hf_model = AutoModel.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
                
        linear = nn.Linear(hf_model.config.hidden_size,colbert_dim,bias=True)
    
        if os.path.exists(os.path.join(model_name_or_path, 'colbert_linear.pt')):
            print('loading colbert_linear weight')
            colbert_state_dict = torch.load(os.path.join(model_name_or_path, 'colbert_linear.pt'), map_location='cpu')
            linear.load_state_dict(colbert_state_dict)



        colbert = cls(hf_model,linear, tokenizer, cuda_device, mask_punctuation, temperature)

        return colbert

def test_relecance():
    ckpt_path=''
    device = 'cuda:0'

    colbert = ColBERT.from_pretrained(
        ckpt_path,
        colbert_dim=1024,
        cuda_device=device
        )
    colbert.eval()
    colbert.to(device)
    input_lst=[
        ['我喜欢中国','我喜欢中国'],
        ['我喜欢中国','我一点都不喜欢中国'],
        ['泰山要多长时间爬上去','爬上泰山需要1-8个小时，具体的时间需要看个人的身体素质。专业登山运动员可能只需要1个多小时就可以登顶，有些身体素质比较低的，爬的慢的就需要5个多小时了。']]
    
    # input_lst=[['What is BGE M3?','BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.']]

    # input_lst=[['What is BGE M3?','BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document']]

    res=colbert.compute_score(input_lst)
    print(res)


if __name__ == "__main__":

    test_relecance()


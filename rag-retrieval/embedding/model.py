
import torch
import torch.nn as nn
import numpy as np
from tqdm import trange


from sentence_transformers import SentenceTransformer
from functools import cached_property
from transformers import AutoTokenizer
import torch.nn.functional as F
import subprocess


class Embedding(nn.Module):
    def __init__(
        self,
        sentence_model = None,
        tokenizer = None,
        temperature = 0.02,
    ):
        super().__init__()


        self.model = sentence_model
        self.tokenizer = tokenizer
        self.temperature = temperature


    def get_embedding(self,input_ids,attention_mask):

        batch_text = {'input_ids': input_ids, 'attention_mask': attention_mask}
        embedding = self.model(batch_text)['sentence_embedding']
        embedding = F.normalize(embedding,p=2,dim=-1)
                
        return embedding    


    def forward(self,
        query_input_ids, # [batch_size,seq_len]
        query_attention_mask, # [batch_size,seq_len]

        pos_doc_input_ids = None, # [batch_size,seq_len]
        pos_doc_attention_mask = None, # [batch_size,seq_len]
        
        neg_doc_input_ids = None, # [batch_size*neg_nums,seq_len]
        neg_doc_attention_mask = None, # [batch_size*neg_nums,seq_len]
    ):
        query_embeddings = self.get_embedding(query_input_ids,query_attention_mask)

        res_dict = {}
        res_dict['query_embeddings'] = query_embeddings
        
        #only pos pair loss
        if pos_doc_input_ids is not None and neg_doc_input_ids is None :  

            pos_doc_embeddings = self.get_embedding(pos_doc_input_ids,pos_doc_attention_mask)
            res_dict['loss'] = self.pair_inbatch_softmax_loss(query_embeddings,pos_doc_embeddings)

        # both pos and neg triplet loss
        elif pos_doc_input_ids is not None and neg_doc_input_ids is not None:

            pos_doc_embeddings = self.get_embedding(pos_doc_input_ids,pos_doc_attention_mask)
            neg_doc_embeddings = self.get_embedding(neg_doc_input_ids,neg_doc_attention_mask)
            res_dict['loss'] = self.triplet_inbatch_softmax_loss(query_embeddings,pos_doc_embeddings,neg_doc_embeddings )

        return res_dict

    def pair_inbatch_softmax_loss(self,
        query_embeddings,
        pos_doc_embeddings,    
    ):
        
        loss_fct = nn.CrossEntropyLoss()
        #[batch_size,batch_size]=[batch_size,dim],[dim,batch_size]
        sim_matrix = query_embeddings @ pos_doc_embeddings.unsqueeze.transpose(-1,-2)
        sim_matrix = sim_matrix / self.temperature
        #[batch_size]
        labels = torch.arange(query_embeddings.size(0), device=query_embeddings.device, dtype=torch.long)
        loss = loss_fct(sim_matrix, labels)
        return loss
    
    def triplet_inbatch_softmax_loss(self,
        query_embeddings,   #[batch_size,dim]
        pos_doc_embeddings, #[batch_size,dim]
        neg_doc_embeddings, #[batch_size*neg_nums,dim]
    ):
        loss_fct = nn.CrossEntropyLoss()

        #[batch_size] <- [batch_size,dim],[batch_size,dim]        
        pos_sim_matrix = torch.sum(query_embeddings * pos_doc_embeddings, dim=-1)

        #[batch_size,1,batch_size*neg_nums] <- [batch_size,1,dim],[1,batch_size*neg_nums,dim]
        neg_sim_matrix = query_embeddings.unsqueeze(1) @ neg_doc_embeddings.unsqueeze(0).transpose(-1,-2)


        #[batch_size,batch_size*neg_nums]
        neg_sim_matrix = neg_sim_matrix.squeeze(1)
        labels = torch.zeros(query_embeddings.shape[0], dtype=torch.long, device=query_embeddings.device) 

        #[batch_size,1+batch_size*neg_nums]
        pos_neg_score = torch.cat([pos_sim_matrix.unsqueeze(1), neg_sim_matrix], dim=1)/self.temperature

        loss = loss_fct(pos_neg_score,labels)

        return loss

    def encode(self,
        sentences,
        device = 'cpu',
        max_len = 512,
        batch_size = 512,
    ):
        self.device = device
        self.to(self.device)
        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches"):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]

            input_ids, token_type_ids, attention_mask= self.preprocess(sentences_batch,max_len)
            
            embeddings=self.forward(input_ids, attention_mask)
            embeddings = embeddings['query_embeddings'].detach().cpu()

            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if len(all_embeddings):
            all_embeddings = torch.stack(all_embeddings)
        else:
            all_embeddings = torch.Tensor()

        return all_embeddings

    def preprocess(self,
        sentences,
        max_len = 512  
    ):
        
        tokens = self.tokenizer(sentences,return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
        

        input_ids = tokens["input_ids"].to(self.device)
        token_type_ids = tokens['token_type_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)

        return input_ids, token_type_ids, attention_mask

    def _text_length(self, text):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):              #{key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):      #Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):    #Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])      #Sum of length of individual strings
    
    def save_pretrained(
        self,
        save_dir
    ):
        def _trans_state_dict(state_dict):
            state_dict = type(state_dict)(
                {k: v.clone().cpu()
                    for k,
                    v in state_dict.items()})
            return state_dict  
        
        self.model.save(save_dir)
        
        #for solve a bug,https://github.com/huggingface/transformers/issues/27293.
        #later upgrade sentence_transformers  will be solve https://github.com/UKPLab/sentence-transformers/issues/2533
        merge_command = f"rm  {save_dir}/model.safetensors"
        subprocess.run(merge_command, shell=True)

        model_state_dict=_trans_state_dict(self.model._modules['0'].state_dict())
        new_model_state_dict={}
        for key,value, in model_state_dict.items():
            new_model_state_dict[key.replace('auto_model.','')]=model_state_dict[key]

        torch.save(new_model_state_dict,save_dir+'/pytorch_model.bin')

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        temperature = 0.02,
    ):
        sentence_model = SentenceTransformer(model_name_or_path)

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        embedding = cls(sentence_model, tokenizer,temperature)

        return embedding

def test_model_embedding():


    cuda_device='cuda:0'
    ckpt_path = ''
    embedding = Embedding.from_pretrained(
        ckpt_path,
    )

    embedding.to(cuda_device)
    input_lst = ['我喜欢中国']
    
    embedding = embedding.encode(input_lst,device=cuda_device)

    print(embedding.tolist())

if __name__ == "__main__":
    test_model_embedding()

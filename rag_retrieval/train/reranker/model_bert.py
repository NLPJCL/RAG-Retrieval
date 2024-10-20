
import torch
from torch import nn
from torch.nn import MSELoss, BCEWithLogitsLoss
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class CrossEncoder(nn.Module):
    def __init__(
        self,
        hf_model = None,
        tokenizer = None,
        cuda_device = 'cpu', 
        loss_type = 'classfication',
    ):
        super().__init__()

        self.model = hf_model
        self.tokenizer = tokenizer
        self.cuda_device=cuda_device
        self.loss_type=loss_type

    def forward(self, batch, labels = None):
        
        output = self.model(**batch,labels=labels)

        if labels is not None:

            logits = output.logits
            if self.loss_type=='regression_mse':
                logits = torch.sigmoid(logits)
                loss_fct = MSELoss()
                loss = loss_fct(logits.squeeze(),labels.squeeze())
            elif self.loss_type=='regression_ce':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.squeeze(),labels.squeeze())
            elif self.loss_type=='classfication':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.squeeze(),labels.squeeze())
            output.loss=loss

        return output
    
    @torch.no_grad()
    def compute_score(
        self,
        sentences_pairs,
        batch_size=256,
        max_len=512,
    ):
        '''
            sentences_pairs=[[query,title],[query1,title1],...]
        '''

        all_logits=[]
        for start_index in range(0, len(sentences_pairs), batch_size):

            sentences_batch=sentences_pairs[start_index:start_index+batch_size]
            batch_data = self.preprocess(sentences_batch,max_len)
            output=self.forward(batch_data)
            logits = output.logits.detach().cpu()
            all_logits.extend(logits)
        return all_logits

    def preprocess(self, sentences,max_len):
        
        tokens = self.tokenizer.batch_encode_plus(sentences,add_special_tokens=True,padding='max_length',truncation=True,
                                    max_length=max_len,return_tensors='pt')

        tokens["input_ids"] = tokens["input_ids"].to(self.cuda_device)
        if 'token_type_ids' in tokens:
            tokens['token_type_ids'] = tokens['token_type_ids'].to(self.cuda_device)
        tokens['attention_mask'] = tokens['attention_mask'].to(self.cuda_device)

        return tokens

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        loss_type = 'classfication',
        num_labels = 1,
        cuda_device='cpu',
    ):
        hf_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        reranker = cls(hf_model, tokenizer,cuda_device,loss_type)
        return reranker

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
        
        self.model.save_pretrained(save_dir,state_dict=_trans_state_dict(self.model.state_dict()),safe_serialization=False)


def test_relecance():
    ckpt_path='hfl/chinese-roberta-wwm-ext'
    device = 'cuda:0'
    cross_encode=CrossEncoder.from_pretrained(ckpt_path,num_labels=1,cuda_device=device)
    cross_encode.eval()
    cross_encode.model.to(device)

    input_lst=[
        ['我喜欢中国','我喜欢中国'],
        ['我喜欢美国','我一点都不喜欢美国'],
        ['泰山要多长时间爬上去','爬上泰山需要1-8个小时，具体的时间需要看个人的身体素质。专业登山运动员可能只需要1个多小时就可以登顶，有些身体素质比较低的，爬的慢的就需要5个多小时了。']]
    
    res=cross_encode.compute_score(input_lst)

    print(torch.sigmoid(res[0]))
    print(torch.sigmoid(res[1]))
    print(torch.sigmoid(res[2]))



if __name__ == "__main__":
    test_relecance()

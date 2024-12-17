
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm

class LLMGenerateDecoder(nn.Module):
    def __init__(
        self,
        hf_model = None,
        tokenizer = None, 
        cuda_device = 'cpu', 
        loss_type = 'point_ce',
    ):
        super().__init__()

        self.model = hf_model
        self.tokenizer = tokenizer
        self.cuda_device = cuda_device
        self.loss_type = loss_type
        self.prompt = """给定一个查询A和一个段落B，请根据段落内容判断该段落是否包含查询的答案，并给出预测结果：“是” 或 “否”。
                        查询A:{query_1}
                        段落B:{passage}
                        """
        self.yes_loc = tokenizer('是', add_special_tokens=False)['input_ids'][0]

    def forward(self, batch, labels = None):
        
        generated_ids_batch = self.model.generate(
            **batch,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True
        )

        return generated_ids_batch
    
    @torch.no_grad()
    def compute_score(
        self,
        sentences_pairs,
        batch_size=12,
        max_len=512,
    ):
        '''
            sentences_pairs=[[query,title],[query1,title1],...]
        '''
        all_logits=[]
        for start_index in tqdm.tqdm(range(0, len(sentences_pairs), batch_size)):

            sentences_batch=sentences_pairs[start_index:start_index+batch_size]
            batch_data = self.preprocess(sentences_batch,max_len)
            generated_ids_batch = self.forward(batch_data)
            generate_prob = torch.softmax(generated_ids_batch.logits[0],-1)
            generate_prob = generate_prob[:,self.yes_loc].view(-1, ).cpu().float().tolist()
            all_logits.extend(generate_prob)
        return all_logits

    def preprocess(self, sentences,max_len):
                
        message_batch = []

        for query, passage in sentences:

            prompt = self.prompt.format(query_1=query, passage=passage)
            temp_dic = {"role": "user", "content": prompt}
            message_batch.append([temp_dic])
        
        text_batch = self.tokenizer.apply_chat_template(
            message_batch,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs_batch = self.tokenizer(text_batch, return_tensors="pt", padding=True).to(self.model.device)

        return model_inputs_batch
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        loss_type = 'point_ce',
        num_labels = 1,
        cuda_device='cpu',
    ):
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
            device_map="balanced_low_0",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="left",
            trust_remote_code=True,
        )
        
        reranker = cls(hf_model, tokenizer,cuda_device,loss_type)
        return reranker

def test_relecance():
    ckpt_path='./Qwen2-1.5B-Instruct'
    llmreranker = LLMGenerateDecoder.from_pretrained(ckpt_path)
    llmreranker.eval()
    input_lst=[
    ['鹦鹉吃自己的小鱼吗','关注养鱼老道,关注更多观赏鱼实践知识,让我们简单养水、轻松养鱼!看来是我错怪了这对迷你鹦鹉鱼,极有可能是我当天看错了,人家本来是不吃孩子的,被我误认为吃了孩子,所以硬生生的给人家分了家。'],
    ['我喜欢美国','我不喜欢美国'],
    ['泰山要多长时间爬上去','爬上泰山需要1-8个小时，具体的时间需要看个人的身体素质。专业登山运动员可能只需要1个多小时就可以登顶，有些身体素质比较低的，爬的慢的就需要5个多小时了。'],
    ['鹦鹉吃自己的小鱼吗','鸮鹦鹉主要是草食性,原生的植物、种子、果实及花粉等,甚至是一些树木的边材都能成为它们的食物。在1984年的一次关于鸮鹦鹉的食物及食性研究中确认了共25种的食物,并证明了它们是一种广泛的草食性生物,对于不同的乔木、灌木以及蕨类植物均感兴趣。鸮鹦鹉的喙能有效地碾磨食物,因此它们只有一个相对小的沙囊,此外,鸮鹦鹉的前肠内有细菌协助发酵及消化植物。另外它们有一套独特的习性,就是会用喙将叶片或蕨叶最具营养的部份挑选出来,难以消化的纤维部份则会留下。']
    ]

    res=llmreranker.compute_score(input_lst)

    print(res)



if __name__ == "__main__":
    test_relecance()


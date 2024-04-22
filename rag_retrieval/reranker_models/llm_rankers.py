from typing import Union, List, Optional, Tuple
from .ranker import BaseRanker
from .result import RankedResults, Result
from copy import deepcopy
from .utils import get_device,get_dtype,vprint
import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class llmreranker(BaseRanker):

    def __init__(self, 
        model_name_or_path: str, 
        dtype: str = None,
        device: str = None,
        verbose: int = 1,
    ):

        self.verbose = verbose
        self.model_name_or_path = model_name_or_path
        self.device = get_device(device, verbose=self.verbose)
        self.dtype = get_dtype(dtype, device=self.device, verbose=self.verbose)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=self.dtype
        ).to(self.device)

        self.model.eval()

        self.yes_loc = self.tokenizer('Yes', add_special_tokens=False)['input_ids'][0]

    @torch.no_grad()
    def compute_score(self, 
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: int = 16,
        max_length: int = 512,
        normalize: bool = False,
        prompt: str = None,
        enable_tqdm: bool = True,
    ):
        
        all_scores = []
        for start_index in tqdm.tqdm(range(0, len(sentence_pairs), batch_size), desc="Compute Scores",
                                    disable=not enable_tqdm):
            sentences_batch = sentence_pairs[start_index:start_index + batch_size]

            inputs = self.get_inputs(sentences_batch,prompt,max_length)
            scores = self.model(**inputs).logits[:, -1, self.yes_loc].view(-1, ).float()
            if 'bce' in self.model_name_or_path or normalize==True:
                scores = torch.sigmoid(scores)
            all_scores.extend(scores.cpu().numpy().tolist())

        if len(all_scores) == 1:
            return all_scores[0]
        return all_scores

    @torch.no_grad()
    def rerank(self, 
        query: str, 
        docs: Union[List[str], str] = None,
        batch_size: int = 256,
        max_length: int = 512,
        long_doc_process_strategy: str="max_score_slice",#['max_score_slice','max_length_truncation']
    ):  
        pass

    def get_inputs(self,
        pairs, 
        prompt=None, 
        max_length=1024):
        if prompt is None:
            prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
        sep = "\n"
        prompt_inputs = self.tokenizer(prompt,
                                return_tensors=None,
                                add_special_tokens=False)['input_ids']
        sep_inputs = self.tokenizer(sep,
                            return_tensors=None,
                            add_special_tokens=False)['input_ids']
        inputs = []
        for query, passage in pairs:
            query_inputs = self.tokenizer(f'A: {query}',
                                    return_tensors=None,
                                    add_special_tokens=False,
                                    max_length=max_length * 3 // 4,
                                    truncation=True)
            passage_inputs = self.tokenizer(f'B: {passage}',
                                    return_tensors=None,
                                    add_special_tokens=False,
                                    max_length=max_length,
                                    truncation=True)
            item = self.tokenizer.prepare_for_model(
                [self.tokenizer.bos_token_id] + query_inputs['input_ids'],
                sep_inputs + passage_inputs['input_ids'],
                truncation='only_second',
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False
            )
            item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
            item['attention_mask'] = [1] * len(item['input_ids'])
            inputs.append(item)
        return self.tokenizer.pad(
                inputs,
                padding=True,
                max_length=max_length + len(sep_inputs) + len(prompt_inputs),
                pad_to_multiple_of=8,
                return_tensors='pt',
        )
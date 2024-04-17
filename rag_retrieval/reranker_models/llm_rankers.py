from ranker import BaseRanker
from typing import Union, List, Optional, Tuple

class llmreranker(BaseRanker):

    def __init__(self, model_name_or_path: str, verbose: int):
        pass
    
    def rerank(self, query: str, docs: List[str], doc_ids: Optional[Union[List[str], str]] = None):
        pass


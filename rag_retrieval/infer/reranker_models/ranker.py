from abc import ABC, abstractmethod
from typing import List, Optional, Union,Tuple

class BaseRanker(ABC):
    @abstractmethod
    def __init__(self, 
        model_name_or_path: str, 
        verbose: int
    ):
        pass
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        docs: Union[List[str],str] = None,
    ):
        """
        reranker docs.
        """
        pass

    @abstractmethod
    def compute_score(
        self,
        sentences_pairs: Union[List[Tuple[str,str]],Tuple[str, str]],
    ):
        """
        compute a list of Tuple[str,str] score or a Tuple[str,str]
        """



AVAILABLE_RANKERS = {}

try:
    from rag_retrieval.reranker_models.cross_encoder_ranker import CorssEncoderRanker
    AVAILABLE_RANKERS["CorssEncoderRanker"] = CorssEncoderRanker
except Exception as e:
    print(e)
    pass

try:
    from rag_retrieval.reranker_models.colbert_ranker import ColBERTRanker

    AVAILABLE_RANKERS["ColBERTRanker"] = ColBERTRanker
except ImportError:
    pass

try:
    from rag_retrieval.reranker_models.api_rankers import APIRanker
    AVAILABLE_RANKERS["APIRanker"] = APIRanker
except ImportError:
    pass

try:
    from rag_retrieval.reranker_models.llm_rankers import LLMRanker

    AVAILABLE_RANKERS["LLMRanker"] = LLMRanker
except ImportError:
    pass



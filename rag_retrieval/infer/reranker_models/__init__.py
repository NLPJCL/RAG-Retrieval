AVAILABLE_RANKERS = {}

try:
    from rag_retrieval.infer.reranker_models.cross_encoder_ranker import CorssEncoderRanker
    AVAILABLE_RANKERS["CorssEncoderRanker"] = CorssEncoderRanker
except Exception as e:
    pass

# try:
#     from rag_retrieval.reranker_models.colbert_ranker import ColBERTRanker

#     AVAILABLE_RANKERS["ColBERTRanker"] = ColBERTRanker
# except Exception as e:
#     pass

# try:
#     from rag_retrieval.reranker_models.api_rankers import APIRanker
#     AVAILABLE_RANKERS["APIRanker"] = APIRanker
# except Exception as e:
#     pass

try:
    from rag_retrieval.infer.reranker_models.llm_rankers import LLMRanker

    AVAILABLE_RANKERS["LLMRanker"] = LLMRanker
except Exception as e:
    pass

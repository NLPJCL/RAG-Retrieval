
import os
os.environ['CUDA_VISIBLE_DEVICES']='7'

from rag_retrieval import Reranker




def test_rag_retrieval_gemma(query,docs):
    
    from rag_retrieval import Reranker
    model_name_or_path='./bge-reranker-v2-gemma/models--BAAI--bge-reranker-v2-gemma/snapshots/1787044f8b6fb740a9de4557c3a12377f84d9e17'

    ranker = Reranker(model_name_or_path,dtype='fp16',verbose=1)

    sentence_pairs= [ [query,doc]  for doc in docs]


    scores = ranker.compute_score(sentence_pairs)


    doc_ranked = ranker.rerank(query,docs)

    return scores,doc_ranked


def test_bge_gemma(query,docs):
    from FlagEmbedding import LayerWiseFlagLLMReranker,FlagLLMReranker
    
    reranker = FlagLLMReranker('./bge-reranker-v2-gemma/models--BAAI--bge-reranker-v2-gemma/snapshots/1787044f8b6fb740a9de4557c3a12377f84d9e17', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    
    sentence_pairs= [ [query,doc]  for doc in docs]

    scores = reranker.compute_score(sentence_pairs,use_dataloader=False)
    
    return scores


query='what is panda?'

docs=['hi','The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']

rag_retrieval_scores,rag_retrieval_doc_ranked=test_rag_retrieval_gemma(query,docs)

bge_scores = test_bge_gemma(query,docs)

print(f'rag_retrieval_doc_ranked is {rag_retrieval_doc_ranked}')

print(f'rag_retrieval_scores is {rag_retrieval_scores}')

print(f'bge_scores is {bge_scores}')

assert rag_retrieval_scores == bge_scores


'''output fp32
rag_retrieval_doc_ranked is results=[Result(doc_id=1, text='The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.', score=10.731430053710938, rank=1), Result(doc_id=0, text='hi', score=-0.9602651000022888, rank=2)] query='what is panda?' has_scores=True
rag_retrieval_scores is [-0.9602651000022888, 10.731430053710938]
bge_scores is [-0.9602651000022888, 10.731430053710938]
'''


'''output fp16
rag_retrieval_doc_ranked is results=[Result(doc_id=1, text='The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.', score=10.703125, rank=1), Result(doc_id=0, text='hi', score=-0.95458984375, rank=2)] query='what is panda?' has_scores=True
rag_retrieval_scores is [-0.95458984375, 10.703125]
bge_scores is [-0.95458984375, 10.703125]
'''

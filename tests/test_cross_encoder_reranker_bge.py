
import os
os.environ['CUDA_VISIBLE_DEVICES']='7'

from rag_retrieval import Reranker




def test_rag_retrieval_cross_encode(query,docs):
    
    from rag_retrieval import Reranker

    model_name_or_path='BAAI/bge-reranker-base'

    ranker = Reranker(model_name_or_path,verbose=1)

    sentence_pairs= [ [query,doc]  for doc in docs]


    scores = ranker.compute_score(sentence_pairs)


    doc_ranked = ranker.rerank(query,docs)

    return scores,doc_ranked


def test_bge_cross_encode(query,docs):

    from FlagEmbedding import FlagReranker
    reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=False) # Setting use_fp16 to True speeds up computation with a slight performance degradation

    sentence_pairs= [ [query,doc]  for doc in docs]


    scores = reranker.compute_score(sentence_pairs)
    
    return scores


query='what is panda?'

docs=['hi','The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']

rag_retrieval_scores,rag_retrieval_doc_ranked=test_rag_retrieval_cross_encode(query,docs)

bge_scores = test_bge_cross_encode(query,docs)

print(f'rag_retrieval_doc_ranked is {rag_retrieval_doc_ranked}')

print(f'rag_retrieval_scores is {rag_retrieval_scores}')

print(f'bge_scores is {bge_scores}')

assert rag_retrieval_scores == bge_scores


'''output fp16
rag_retrieval_doc_ranked is results=[Result(doc_id=1, text='The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.', score=6.18359375, rank=1), Result(doc_id=0, text='hi', score=-8.1484375, rank=2)] query='what is panda?' has_scores=True
rag_retrieval_scores is [-8.1484375, 6.18359375]
bge_scores is [-8.1484375, 6.18359375]
'''

'''output fp32
rag_retrieval_doc_ranked is results=[Result(doc_id=1, text='The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.', score=6.182114124298096, rank=1), Result(doc_id=0, text='hi', score=-8.154389381408691, rank=2)] query='what is panda?' has_scores=True
rag_retrieval_scores is [-8.154389381408691, 6.182114124298096]
bge_scores is [-8.154389381408691, 6.182114124298096]
'''


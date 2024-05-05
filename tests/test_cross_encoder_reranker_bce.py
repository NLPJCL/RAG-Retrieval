
import os
os.environ['CUDA_VISIBLE_DEVICES']='7'

from rag_retrieval import Reranker




def test_rag_retrieval_cross_encode(query,docs):
    
    from rag_retrieval import Reranker

    model_name_or_path='maidalun1020/bce-reranker-base_v1'

    ranker = Reranker(model_name_or_path,verbose=1,dtype='fp16')

    sentence_pairs= [ [query,doc]  for doc in docs]


    scores = ranker.compute_score(sentence_pairs)


    doc_ranked = ranker.rerank(query,docs)

    return scores,doc_ranked


def test_bce_cross_encoder(query,docs):

    #from FlagEmbedding import FlagReranker

    from BCEmbedding import RerankerModel


    
    reranker = RerankerModel('maidalun1020/bce-reranker-base_v1', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

    sentence_pairs= [ [query,doc]  for doc in docs]


    scores = reranker.compute_score(sentence_pairs)
    
    doc_ranked = reranker.rerank(query,docs)


    return scores,doc_ranked


query='what is panda?'

docs=['hi','The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']

rag_retrieval_scores,rag_retrieval_doc_ranked=test_rag_retrieval_cross_encode(query,docs)

bce_scores,bce_doc_ranked = test_bce_cross_encoder(query,docs)

print(f'rag_retrieval_doc_ranked is {rag_retrieval_doc_ranked}')

print(f'rag_retrieval_scores is {rag_retrieval_scores}')

print(f'bce_doc_ranked is {bce_doc_ranked}')

print(f'bce_scores is {bce_scores}')

assert rag_retrieval_scores == bce_scores


'''output fp16
rag_retrieval_doc_ranked is results=[Result(doc_id=1, text='The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.', score=0.5789839029312134, rank=1), Result(doc_id=0, text='hi', score=0.34400254487991333, rank=2)] query='what is panda?' has_scores=True
rag_retrieval_scores is [0.34400254487991333, 0.5789839029312134]
bce_doc_ranked is {'rerank_passages': ['The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.', 'hi'], 'rerank_scores': [0.5789839029312134, 0.34400254487991333], 'rerank_ids': [1, 0]}
bce_scores is [0.34400254487991333, 0.5789839029312134]
'''

'''output fp32
rag_retrieval_doc_ranked is results=[Result(doc_id=1, text='The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.', score=0.5790157914161682, rank=1), Result(doc_id=0, text='hi', score=0.34393033385276794, rank=2)] query='what is panda?' has_scores=True
rag_retrieval_scores is [0.34393033385276794, 0.5790157914161682]
bce_doc_ranked is {'rerank_passages': ['The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.', 'hi'], 'rerank_scores': [0.5790157914161682, 0.34393033385276794], 'rerank_ids': [1, 0]}
bce_scores is [0.34393033385276794, 0.5790157914161682]
'''


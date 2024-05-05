
import os
os.environ['CUDA_VISIBLE_DEVICES']='7'

from rag_retrieval import Reranker




def test_rag_retrieval_cpm(query,docs):
    
    from rag_retrieval import Reranker
    model_name_or_path='./bge-reranker-v2-minicpm-layerwise/models--BAAI--bge-reranker-v2-minicpm-layerwise/snapshots/47b5332b296c4d8cb6ee2c60502cc62a0d708881'

    ranker = Reranker(model_name_or_path,dtype='fp16',verbose=1)

    sentence_pairs= [ [query,doc]  for doc in docs]


    scores = ranker.compute_score(sentence_pairs,cutoff_layers=[28])


    doc_ranked = ranker.rerank(query,docs,cutoff_layers=[28])

    return scores,doc_ranked


def test_bge_cpm(query,docs):
    from FlagEmbedding import LayerWiseFlagLLMReranker,FlagLLMReranker
    
    reranker = LayerWiseFlagLLMReranker('./bge-reranker-v2-minicpm-layerwise/models--BAAI--bge-reranker-v2-minicpm-layerwise/snapshots/47b5332b296c4d8cb6ee2c60502cc62a0d708881', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    
    sentence_pairs= [ [query,doc]  for doc in docs]

    scores = reranker.compute_score(sentence_pairs,use_dataloader=False,cutoff_layers=[28])
    
    return scores


query='what is panda?'

docs=['hi','The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']

rag_retrieval_scores,rag_retrieval_doc_ranked=test_rag_retrieval_cpm(query,docs)

bge_scores = test_bge_cpm(query,docs)

print(f'rag_retrieval_doc_ranked is {rag_retrieval_doc_ranked}')

print(f'rag_retrieval_scores is {rag_retrieval_scores}')

print(f'bge_scores is {bge_scores}')

assert rag_retrieval_scores == bge_scores


'''output fp16,不设置cutoff_layers
rag_retrieval_doc_ranked is results=[Result(doc_id=1, text='The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.', score=1.7685546875, rank=1), Result(doc_id=0, text='hi', score=-9.953125, rank=2)] query='what is panda?' has_scores=True
rag_retrieval_scores is [-9.953125, 1.7685546875]
bge_scores is [-9.953125, 1.7685546875]
'''

'''output fp16,cutoff_layers=[28]
rag_retrieval_doc_ranked is results=[Result(doc_id=1, text='The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.', score=27.453125, rank=1), Result(doc_id=0, text='hi', score=15.1171875, rank=2)] query='what is panda?' has_scores=True
rag_retrieval_scores is [15.1171875, 27.453125]
bge_scores is [15.1171875, 27.453125]
'''


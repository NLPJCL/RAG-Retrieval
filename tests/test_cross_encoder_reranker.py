
import os
os.environ['CUDA_VISIBLE_DEVICES']='7'

from rag_retrieval import Reranker


ranker = Reranker('BAAI/bge-reranker-base',dtype='fp16',verbose=0)

# pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]


# scores = ranker.compute_score(pairs)

# print(scores)


query='what is panda?'

docs=['hi','The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']

doc_ranked = ranker.rerank(query,docs)


print(doc_ranked.results)

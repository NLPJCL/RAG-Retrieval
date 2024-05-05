
import os
os.environ['CUDA_VISIBLE_DEVICES']='7'

from rag_retrieval import Reranker


model_name_or_path='/data/sealgo/user/lijiacheng/bge-reranker-v2-gemma/models--BAAI--bge-reranker-v2-gemma/snapshots/1787044f8b6fb740a9de4557c3a12377f84d9e17'

ranker = Reranker(model_name_or_path,dtype='fp16',verbose=1)

pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]


scores = ranker.compute_score(pairs)

print(scores)


# query='what is panda?'

# docs=['hi','The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']

# doc_ranked = ranker.rerank(query,docs)
# print(doc_ranked)

# print(type(doc_ranked))


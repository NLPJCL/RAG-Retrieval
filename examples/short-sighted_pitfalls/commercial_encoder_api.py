import numpy as np
from abc import ABC
from typing import List
import requests
from sklearn.preprocessing import normalize
import hashlib
import diskcache as dc
from tqdm import tqdm
import time

cache = dc.Cache("./.api_cache")


class CommercialEncoder(ABC):
    def encode(
        self,
        sentences: List[str],
        normalize_embeddings: bool = False,
        *args,
        **kwargs,
    ):
        raise NotImplementedError

    def _generate_cache_key(self, model_name: str, text: str, **kwargs):
        """生成哈希值作为缓存键，确保相同输入得到相同缓存"""
        hash_input = model_name + "_" + text + "_" + str(kwargs)
        return hashlib.md5(hash_input.encode()).hexdigest()


class OpenAIEncoder(CommercialEncoder):
    def __init__(self):
        from openai import OpenAI

        # text-embedding-3-small, 1536d, 8191 tokens
        # text-embedding-3-large, 3072d, 8191 tokens
        self.client = OpenAI()

    def encode(
        self,
        model_name: str = "text-embedding-3-large",
        sentences: List[str] = None,
        normalize_embeddings: bool = False,
        bsz: int = 64,
        *args,
        **kwargs,
    ):
        vectors = []
        for start in tqdm(range(0, len(sentences), bsz)):
            sentences_split = sentences[start : start + bsz]
            sentences_filtered_index = []
            for i,text in enumerate(sentences_split):
                cache_key = self._generate_cache_key(model_name, text, **kwargs)
                if cache_key not in cache:
                    sentences_filtered_index.append(i)
            sentences_split_filtered = [sentences_split[i] for i in sentences_filtered_index]
            
            if len(sentences_split_filtered) > 0:
                embedding_data = self.client.embeddings.create(
                    input=sentences_split_filtered, model=model_name
                ).data
                embedding_data = [item.embedding for item in embedding_data]
                assert len(embedding_data) == len(sentences_filtered_index)
                for i, index in enumerate(sentences_filtered_index):
                    cache_key = self._generate_cache_key(model_name, sentences_split[index], **kwargs)
                    cache[cache_key] = embedding_data[i]
            
            embedding_data = []
            for i, text in enumerate(sentences_split):
                cache_key = self._generate_cache_key(model_name, text, **kwargs)
                try:
                    embedding_data.append(cache[cache_key])
                except KeyError:
                    print(f"KeyError: {cache_key}, {text}")
                
            vectors.extend(embedding_data)
            
        vectors = np.array(vectors, dtype=np.float32)
        if normalize_embeddings:
            vectors = normalize(vectors)

        return vectors

    @staticmethod
    def test():
        openai_encoder = OpenAIEncoder()
        vectors = openai_encoder.encode(
            sentences=["""According to a National Geographic article, the novel is so revered in Monroeville that people quote lines from it like Scripture; yet Harper Lee herself refused to attend any performances, because "she abhors anything that trades on the book's fame". To underscore this sentiment, Lee demanded that a book of recipes named Calpurnia's Cookbook not be published and sold out of the Monroe County Heritage Museum. David Lister in The Independent states that Lee's refusal to speak to reporters made them desire to interview her all the more, and her silence "makes Bob Dylan look like a media tart". Despite her discouragement, a rising number of tourists made to Monroeville a destination, hoping to see Lee's inspiration for the book, or Lee herself. Local residents call them "Mockingbird groupies", and although Lee was not reclusive, she refused publicity and interviews with an emphatic "Hell, no!"""], normalize_embeddings=True
        )
        print(vectors)
        print(vectors.shape)


class CohereEncoder(CommercialEncoder):
    def __init__(self):
        import cohere
        # embed-english-v3.0, 1024d, 512 tokens best
        # embed-multilingual-v3.0, 1024d, 512 tokens best
        self.client = cohere.ClientV2()

    def encode(
        self,
        model_name: str = "embed-english-v3.0",
        sentences: List[str] = None,
        bsz: int = 64,
        normalize_embeddings=False,
        *args,
        **kwargs,
    ):
        vectors = []
        for start in tqdm(range(0, len(sentences), bsz)):
            sentences_split = sentences[start : start + bsz]
            sentences_filtered_index = []
            for i,text in enumerate(sentences_split):
                cache_key = self._generate_cache_key(model_name, text, **kwargs)
                if cache_key not in cache:
                    sentences_filtered_index.append(i)
            sentences_split_filtered = [sentences_split[i] for i in sentences_filtered_index]
            
            if len(sentences_split_filtered) > 0:
                embedding_data = self.client.embed(
                    texts=sentences_split_filtered,
                    model=model_name,
                    input_type=kwargs.get("prompt_name"),
                    embedding_types=["float"],
                    truncate="END",
                ).embeddings.float
                for i, index in enumerate(sentences_filtered_index):
                    cache_key = self._generate_cache_key(model_name, sentences_split[index], **kwargs)
                    cache[cache_key] = embedding_data[i]
            
            embedding_data = []
            for i, text in enumerate(sentences_split):
                cache_key = self._generate_cache_key(model_name, text, **kwargs)
                embedding_data.append(cache[cache_key])
                
            vectors.extend(embedding_data)
            
        vectors = np.array(vectors, dtype=np.float32)
        if normalize_embeddings:
            vectors = normalize(vectors)

        return vectors

    @staticmethod
    def test():
        cohere_encoder = CohereEncoder()
        vectors = cohere_encoder.encode(
            sentences=["hi", "a sentence", "nice to see you"],
            normalize_embeddings=True,
            prompt_name="search_document",  # search_document, search_query
        )
        print(vectors)
        print(vectors.shape)  # (3, 1024)


class VoyageEncoder(CommercialEncoder):
    def __init__(self):
        import voyageai

        # voyage-3-m-exp, 2048d(MRL defalut 1024d), 32k tokens, 6918M
        self.client = voyageai.Client()

    def encode(
        self,
        model_name: str = "voyage-3-large", # "voyage-3-m-exp"
        bsz: int = 64,
        sentences: List[str] = None,
        normalize_embeddings=False,
        *args,
        **kwargs,
    ):
        vectors = []
        for start in tqdm(range(0, len(sentences), bsz)):
            sentences_split = sentences[start : start + bsz]
            sentences_filtered_index = []
            for i,text in enumerate(sentences_split):
                cache_key = self._generate_cache_key(model_name, text, **kwargs)
                if cache_key not in cache:
                    sentences_filtered_index.append(i)
            sentences_split_filtered = [sentences_split[i] for i in sentences_filtered_index]
            
            if len(sentences_split_filtered) > 0:
                embedding_data = self.client.embed(
                    texts=sentences_split_filtered,
                    model=model_name,
                    input_type=kwargs.get("prompt_name", "document"),
                    truncation=True,
                    output_dtype="float",
                    output_dimension=kwargs.get("output_dimension", 1024),
                ).embeddings
                for i, index in enumerate(sentences_filtered_index):
                    cache_key = self._generate_cache_key(model_name, sentences_split[index], **kwargs)
                    cache[cache_key] = embedding_data[i]
            
            embedding_data = []
            for i, text in enumerate(sentences_split):
                cache_key = self._generate_cache_key(model_name, text, **kwargs)
                embedding_data.append(cache[cache_key])
                
            vectors.extend(embedding_data)
            
        vectors = np.array(vectors, dtype=np.float32)
        if normalize_embeddings:
            vectors = normalize(vectors)

        return vectors

    @staticmethod
    def test():
        voyage_encoder = VoyageEncoder()
        vectors = voyage_encoder.encode(
            sentences=["hi", "a sentence", "nice to see you"],
            normalize_embeddings=True,
            prompt_name="document",  # document, query
            output_dimension=2048,
        )
        print(vectors)
        print(vectors.shape)  # (3, 2048)


class JinaEncoder(CommercialEncoder):
    def __init__(self):
        # jina-embeddings-v3, 1024d, 8192 tokens, 559M
        self.api_key = ""

    def encode(
        self,
        model_name: str = "jina-embeddings-v3",
        bsz: int = 64,
        sentences: List[str] = None,
        normalize_embeddings: bool = False,
        *args,
        **kwargs,
    ):
        url = "https://api.jina.ai/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = {
            "model": f"{model_name}",
            "task": kwargs.get("prompt_name", "retrieval.passage"),  # retrieval.query
            "late_chunking": False,
            "dimensions": kwargs.get("output_dimension", 1024),
            "embedding_type": "float",
        }
        
        vectors = []
        for start in tqdm(range(0, len(sentences), bsz)):
            sentences_split = sentences[start : start + bsz]
            sentences_filtered_index = []
            for i,text in enumerate(sentences_split):
                cache_key = self._generate_cache_key(model_name, text, **kwargs)
                if cache_key not in cache:
                    sentences_filtered_index.append(i)
            sentences_split_filtered = [sentences_split[i] for i in sentences_filtered_index]
            data["input"] = sentences_split_filtered
            if len(sentences_split_filtered) > 0:
                cnt = 1
                while True:
                    try:
                        embedding_data = requests.post(url, headers=headers, json=data).json()["data"]
                        break
                    except Exception as e:
                        cnt += 1
                        print(f"{e}:Retry {cnt} times")
                        time.sleep(2)
                embedding_data = [item["embedding"] for item in embedding_data]
                for i, index in enumerate(sentences_filtered_index):
                    cache_key = self._generate_cache_key(model_name, sentences_split[index], **kwargs)
                    cache[cache_key] = embedding_data[i]
            
            embedding_data = []
            for i, text in enumerate(sentences_split):
                cache_key = self._generate_cache_key(model_name, text, **kwargs)
                embedding_data.append(cache[cache_key])
                
            vectors.extend(embedding_data)
            
        vectors = np.array(vectors, dtype=np.float32)
        if normalize_embeddings:
            vectors = normalize(vectors)

        return vectors

    @staticmethod
    def test():
        jina_encoder = JinaEncoder()
        vectors = jina_encoder.encode(
            sentences=["hi", "a sentence", "nice to see you"],
            normalize_embeddings=True,
            prompt_name="retrieval.passage",  # retrieval.passage, retrieval.query
            output_dimension=1024,
        )
        print(vectors)
        print(vectors.shape)


if __name__ == "__main__":
    OpenAIEncoder.test()
    # CohereEncoder.test()
    # VoyageEncoder.test()
    # JinaEncoder.test()

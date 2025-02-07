import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np
import faiss
import commercial_encoder_api


def find_topk_by_vecs(source_vecs: np.ndarray, target_vecs: np.ndarray, topk: int):
    faiss_index = faiss.IndexFlatIP(target_vecs.shape[1])
    faiss_index.add(target_vecs)

    res_distance, res_index = faiss_index.search(source_vecs, topk)
    return res_index, res_distance


if __name__ == "__main__":
    data_cache_dir = ""
    model_cache_dir = ""

    model_name = "Alibaba-NLP/gte-Qwen2-7B-instruct"
    # model_name = "api_voyage" # "api_openai", "api_cohere", "api_voyage", "api_jina"
    dataset_name = "rajpurkar/squad_v2"

    topk_list = [1, 5, 10, 20, 30, 50]
    batch_size = 16

    # load and process data
    query_answer_start_list, passage_list, query2passage = [], [], {}
    for item in (
        load_dataset(dataset_name, split="train", cache_dir=data_cache_dir).to_list()
        + load_dataset(
            dataset_name, split="validation", cache_dir=data_cache_dir
        ).to_list()
    ):
        # if the answer of question is in the document, then the document is a positive document
        if item["answers"]["answer_start"]:
            query_answer_start_list.append(
                [item["question"], item["answers"]["answer_start"][0]]
            )
            passage_list.append(item["context"])
            query2passage[item["question"]] = item["context"]
        else:
            passage_list.append(item["context"])
    passage_list = list(set(passage_list))
    passage2id = {passage: idx for idx, passage in enumerate(passage_list)}
    labels = np.array(
        [[passage2id[query2passage[query]]] for query, _ in query_answer_start_list]
    )
    answer_start_list = [answer_start for _, answer_start in query_answer_start_list]
    print("min(answer_start_list)", min(answer_start_list))
    print("max(answer_start_list)", max(answer_start_list))
    print("number of all queries", len(query_answer_start_list))
    print("number of all passages", len(passage_list))
    print("min len of passage (words): ", min([len(passage.split(" ")) for passage in passage_list]))
    print("max len of passage (words): ", max([len(passage.split(" ")) for passage in passage_list]))
    # load model
    if "api" not in model_name:
        model = SentenceTransformer(
            model_name, trust_remote_code=True, cache_folder=model_cache_dir
        )
    else:
        model = None
    print(model_name)

    # get q,p vecs
    if "jina-embeddings-v3" in model_name:
        model.max_seq_length = 8192
        q_vecs = model.encode(
            [item[0] for item in query_answer_start_list],
            task="retrieval.query",
            prompt_name="retrieval.query",
            show_progress_bar=True,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
        p_vecs = model.encode(
            passage_list,
            task="retrieval.passage",
            prompt_name="retrieval.passage",
            show_progress_bar=True,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
    elif "bge-m3" in model_name or "jina-embeddings-v2-base-en" in model_name:
        model.max_seq_length = 8192
        q_vecs = model.encode(
            [item[0] for item in query_answer_start_list],
            show_progress_bar=True,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
        p_vecs = model.encode(
            passage_list,
            show_progress_bar=True,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
    elif "jasper_en_vision_language_v1" in model_name:
        model.max_seq_length = 8192
        q_vecs = model.encode(
            [item[0] for item in query_answer_start_list],
            show_progress_bar=True,
            prompt_name="s2p_query",
            batch_size=batch_size,
            normalize_embeddings=True,
        )
        p_vecs = model.encode(
            passage_list,
            show_progress_bar=True,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
    elif "nvidia" in model_name:
        # Each query needs to be accompanied by an corresponding instruction describing the task.
        task_name_to_instruct = {
            "example": "Given a question, retrieve passages that answer the question",
        }
        query_prefix = "Instruct: " + task_name_to_instruct["example"] + "\nQuery: "

        def add_eos(input_examples):
            input_examples = [
                input_example + model.tokenizer.eos_token
                for input_example in input_examples
            ]
            return input_examples

        model.max_seq_length = 32768
        model.tokenizer.padding_side = "right"
        q_vecs = model.encode(
            add_eos([item[0] for item in query_answer_start_list]),
            show_progress_bar=True,
            batch_size=batch_size,
            normalize_embeddings=True,
            prompt=query_prefix,
        )
        p_vecs = model.encode(
            add_eos(passage_list),
            show_progress_bar=True,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
    elif "gte-Qwen2" in model_name:
        model.max_seq_length = 8192
        q_vecs = model.encode(
            [item[0] for item in query_answer_start_list],
            show_progress_bar=True,
            batch_size=batch_size,
            normalize_embeddings=True,
            prompt_name="query",
        )
        p_vecs = model.encode(
            passage_list,
            show_progress_bar=True,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
    elif "api" in model_name:
        model_name = model_name.split("_")[1]
        if model_name == "openai":
            encoder = commercial_encoder_api.OpenAIEncoder()
            q_vecs = encoder.encode(
                sentences=[item[0] for item in query_answer_start_list],
                normalize_embeddings=True,
            )
            p_vecs = encoder.encode(
                sentences=passage_list,
                normalize_embeddings=True,
            )
        elif model_name == "cohere":
            encoder = commercial_encoder_api.CohereEncoder()
            q_vecs = encoder.encode(
                sentences=[item[0] for item in query_answer_start_list],
                normalize_embeddings=True,
                prompt_name="search_query",
            )
            p_vecs = encoder.encode(
                sentences=passage_list,
                normalize_embeddings=True,
                prompt_name="search_document",
            )
        elif model_name == "voyage":
            encoder = commercial_encoder_api.VoyageEncoder()
            q_vecs = encoder.encode(
                sentences=[item[0] for item in query_answer_start_list],
                normalize_embeddings=True,
                prompt_name="query",
                output_dimension=2048,
            )
            p_vecs = encoder.encode(
                sentences=passage_list,
                normalize_embeddings=True,
                prompt_name="document",
                output_dimension=2048,
            )

        elif model_name == "jina":
            encoder = commercial_encoder_api.JinaEncoder()
            q_vecs = encoder.encode(
                sentences=[item[0] for item in query_answer_start_list],
                normalize_embeddings=True,
                prompt_name="retrieval.query",
                output_dimension=1024,
            )
            p_vecs = encoder.encode(
                sentences=passage_list,
                normalize_embeddings=True,
                prompt_name="retrieval.passage",
                output_dimension=1024,
            )
    else:
        raise Exception(f"unsupported model {model_name}")

    # search topk
    topk_index, _ = find_topk_by_vecs(q_vecs, p_vecs, max(topk_list))

    print(
        f"model, #queries, min_answer_start, max_answer_start, {', '.join([f'Recall@{k}' for k in topk_list])}"
    )
    # compute recall with different answer_start and top-k
    for min_len, max_len in [
        (0, 100),
        (100, 200),
        (200, 300),
        (300, 400),
        (400, 500),
        (500, 3120),
    ]:
        recall_at_k_list = []
        selected_ids = [
            idx
            for idx, answer_start in enumerate(answer_start_list)
            if min_len <= answer_start <= max_len
        ]
        for topk in topk_list:
            recall_at_k_list.append(
                (topk_index[selected_ids, :topk] == labels[selected_ids, :]).sum()
                / len(selected_ids)
            )
        recall_at_k_list = [str(float(i)) for i in recall_at_k_list]  # for joining
        print(
            f"{model_name}, {len(selected_ids)}, {min_len}, {max_len}, {','.join(recall_at_k_list)}"
        )

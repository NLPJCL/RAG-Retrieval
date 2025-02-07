import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import random
import tqdm
from datatrove.pipeline.readers import ParquetReader
import commercial_encoder_api

if __name__ == "__main__":
    data_cache_dir = ""
    model_cache_dir = ""
    model_dir = ""

    data_set_name = "fineweb_edu"  # squad_v2, fineweb_edu
    all_models = [
        "jinaai/jina-embeddings-v2-base-en",
        "jinaai/jina-embeddings-v3",
        "BAAI/bge-m3",
        "NovaSearch/jasper_en_vision_language_v1",
        "nvidia/NV-Embed-v2",
        "Alibaba-NLP/gte-Qwen2-7B-instruct",
        "api_voyage",
        "api_openai"
    ]

    max_data = 10000
    # min_len, max_len = 100, 512 # squad_v2
    min_len, max_len = 200, 500  # fineweb_edu
    batch_size = 12

    # load and process data
    text_list = []
    if "squad_v2" in data_set_name:
        for item in (
            load_dataset(
                "rajpurkar/squad_v2", split="train", cache_dir=data_cache_dir
            ).to_list()
            + load_dataset(
                "rajpurkar/squad_v2", split="validation", cache_dir=data_cache_dir
            ).to_list()
        ):
            text_list.append(item["context"])
    elif "fineweb_edu" in data_set_name:
        data_reader = ParquetReader(
            "hf://datasets/HuggingFaceFW/fineweb-edu",
            glob_pattern="sample/10BT/000_00000.parquet",
            limit=10 * 10000,
        )
        for document in tqdm.tqdm(data_reader()):
            text_list.append(document.text)
        # ----
        # fw = load_dataset(
        #     "HuggingFaceFW/fineweb-edu",
        #     name="sample-10BT",
        #     split="train",
        #     streaming=True,
        #     cache_dir=data_cache_dir,
        #     data_files="sample/10BT/000_00000.parquet",
        # )
        # for document in tqdm.tqdm(fw):
        #     text_list.append(document["text"])
        # ----
        # import polars as pl
        # parquet_data = pl.read_parquet("fineweb-edu/000_00000.parquet").rows(named=True)
        # for document in tqdm.tqdm(parquet_data):
        #     text_list.append(document["text"])
    else:
        raise Exception(f"unknown data set name{data_set_name}")
    text_list = [
        passage
        for passage in list(set(text_list))
        if min_len < len(passage.split(" ")) < max_len
    ]
    text_list.sort()
    random.seed(1)
    random.shuffle(text_list)
    text_list = text_list[:max_data]
    print(min_len, max_len)
    print("len(text_list)", len(text_list))
    print(
        "data_set_name",
        "len(text_list)",
        "average_len",
        "model_name_or_path",
        "before_to_full_sim",
        "middle_to_full_sim",
        "after_to_full_sim",
    )

    before_text_list = [passage[: int(len(passage) * 0.333)] for passage in text_list]
    middle_text_list = [
        passage[int(len(passage) * 0.333) : int(len(passage) * 0.666)]
        for passage in text_list
    ]
    after_text_list = [passage[int(len(passage) * 0.666) :] for passage in text_list]

    for model_name_or_path in all_models:
        # load model
        if "api" not in model_name_or_path:
            model = SentenceTransformer(
                model_dir + model_name_or_path, trust_remote_code=True
            )
        else:
            model = None
        print(model_name_or_path)

        # get text vectors
        if "jina-embeddings-v3" in model_name_or_path:
            model.max_seq_length = 8192
            full_text_vecs = model.encode(
                text_list,
                task="retrieval.passage",
                prompt_name="retrieval.passage",
                show_progress_bar=True,
                normalize_embeddings=True,
                batch_size=batch_size,
            )
            before_text_vecs = model.encode(
                before_text_list,
                task="retrieval.passage",
                prompt_name="retrieval.passage",
                show_progress_bar=True,
                normalize_embeddings=True,
                batch_size=batch_size,
            )
            middle_text_vecs = model.encode(
                middle_text_list,
                task="retrieval.passage",
                prompt_name="retrieval.passage",
                show_progress_bar=True,
                normalize_embeddings=True,
                batch_size=batch_size,
            )
            after_text_vecs = model.encode(
                after_text_list,
                task="retrieval.passage",
                prompt_name="retrieval.passage",
                show_progress_bar=True,
                normalize_embeddings=True,
                batch_size=batch_size,
            )
        elif (
            "bge-m3" in model_name_or_path
            or "jina-embeddings-v2-base-en" in model_name_or_path
        ):
            model.max_seq_length = 8192
            full_text_vecs = model.encode(
                text_list,
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
            before_text_vecs = model.encode(
                before_text_list,
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
            middle_text_vecs = model.encode(
                middle_text_list,
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
            after_text_vecs = model.encode(
                after_text_list,
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
        elif "jasper_en_vision_language_v1" in model_name_or_path:
            full_text_vecs = model.encode(
                text_list,
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
            before_text_vecs = model.encode(
                before_text_list,
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
            middle_text_vecs = model.encode(
                middle_text_list,
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
            after_text_vecs = model.encode(
                after_text_list,
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
        elif "nvidia" in model_name_or_path:
            # Each query needs to be accompanied by an corresponding instruction describing the task.
            def add_eos(input_examples):
                input_examples = [
                    input_example + model.tokenizer.eos_token
                    for input_example in input_examples
                ]
                return input_examples

            model.max_seq_length = 32768
            model.tokenizer.padding_side = "right"

            full_text_vecs = model.encode(
                add_eos(text_list),
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
            before_text_vecs = model.encode(
                add_eos(before_text_list),
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
            middle_text_vecs = model.encode(
                add_eos(middle_text_list),
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
            after_text_vecs = model.encode(
                add_eos(after_text_list),
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
        elif "gte-Qwen2" in model_name_or_path:
            model.max_seq_length = 8192
            full_text_vecs = model.encode(
                text_list,
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
            before_text_vecs = model.encode(
                before_text_list,
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
            middle_text_vecs = model.encode(
                middle_text_list,
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
            after_text_vecs = model.encode(
                after_text_list,
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
        elif "api" in model_name_or_path:
            model_name_or_path = model_name_or_path.split("_")[1]
            if model_name_or_path == "openai":
                encoder = commercial_encoder_api.OpenAIEncoder()
                full_text_vecs = encoder.encode(
                    sentences=text_list,
                    normalize_embeddings=True,
                )
                before_text_vecs = encoder.encode(
                    sentences=before_text_list,
                    normalize_embeddings=True,
                )
                middle_text_vecs = encoder.encode(
                    sentences=middle_text_list,
                    normalize_embeddings=True,
                )
                after_text_vecs = encoder.encode(
                    sentences=after_text_list,
                    normalize_embeddings=True,
                )
            elif model_name_or_path == "cohere":
                encoder = commercial_encoder_api.CohereEncoder()
                full_text_vecs = encoder.encode(
                    sentences=text_list,
                    normalize_embeddings=True,
                    prompt_name="search_document",
                )
                before_text_vecs = encoder.encode(
                    sentences=before_text_list,
                    normalize_embeddings=True,
                    prompt_name="search_document",
                )
                middle_text_vecs = encoder.encode(
                    sentences=middle_text_list,
                    normalize_embeddings=True,
                    prompt_name="search_document",
                )
                after_text_vecs = encoder.encode(
                    sentences=after_text_list,
                    normalize_embeddings=True,
                    prompt_name="search_document",
                )
            elif model_name_or_path == "voyage":
                encoder = commercial_encoder_api.VoyageEncoder()
                full_text_vecs = encoder.encode(
                    sentences=text_list,
                    normalize_embeddings=True,
                    prompt_name="document",
                    output_dimension=2048,
                )
                before_text_vecs = encoder.encode(
                    sentences=before_text_list,
                    normalize_embeddings=True,
                    prompt_name="document",
                    output_dimension=2048,
                )
                middle_text_vecs = encoder.encode(
                    sentences=middle_text_list,
                    normalize_embeddings=True,
                    prompt_name="document",
                    output_dimension=2048,
                )
                after_text_vecs = encoder.encode(
                    sentences=after_text_list,
                    normalize_embeddings=True,
                    prompt_name="document",
                    output_dimension=2048,
                )
            elif model_name_or_path == "jina":
                encoder = commercial_encoder_api.JinaEncoder()
                full_text_vecs = encoder.encode(
                    sentences=text_list,
                    normalize_embeddings=True,
                    prompt_name="retrieval.passage",
                    output_dimension=1024,
                )
                before_text_vecs = encoder.encode(
                    sentences=before_text_list,
                    normalize_embeddings=True,
                    prompt_name="retrieval.passage",
                    output_dimension=1024,
                )
                middle_text_vecs = encoder.encode(
                    sentences=middle_text_list,
                    normalize_embeddings=True,
                    prompt_name="retrieval.passage",
                    output_dimension=1024,
                )
                after_text_vecs = encoder.encode(
                    sentences=after_text_list,
                    normalize_embeddings=True,
                    prompt_name="retrieval.passage",
                    output_dimension=1024,
                )
        else:
            raise Exception(f"unsupported model {model_name_or_path}")

        # compute average cosine similarity
        before_to_full_sim = float(
            (full_text_vecs * before_text_vecs).sum(axis=1).mean()
        )
        middle_to_full_sim = float(
            (full_text_vecs * middle_text_vecs).sum(axis=1).mean()
        )
        after_to_full_sim = float((full_text_vecs * after_text_vecs).sum(axis=1).mean())

        print(
            data_set_name,
            len(text_list),
            sum([len(text.split(" ")) for text in text_list]) / len(text_list),
            model_name_or_path,
            before_to_full_sim,
            middle_to_full_sim,
            after_to_full_sim,
        )
        if model:
            del model

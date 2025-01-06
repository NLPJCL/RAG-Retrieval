[English](./README.md) | [中文](./README_zh.md)
# Setting Up the Environment
```bash
conda create -n rag-retrieval python=3.8 && conda activate rag-retrieval
# To avoid compatibility issues between automatically installed torch and local CUDA,
# it is recommended to manually install torch compatible with your local CUDA version before proceeding to the next step.
pip install -r requirements.txt 
```

# Fine-tuning the Model
After installing the dependencies, we will demonstrate how to fine-tune an open-source embedding model using our own data through specific examples.

# Data Format

Similar to bge models, the training data is a jsonl file, with each line formatted as shown below. Here, `pos` is a list of positive document texts, and `neg` is a list of negative document texts,`prompt_for_query` is a prompt that some embedding models need to add before the query. (optional)

For embedding models, the following four types of data are supported for fine-tuning:

- Query and positive documents, where negative examples are randomly sampled from other queries' positive documents within the batch.
```
{"query": str, "pos": List[str], "prompt_for_query"(optional): str}
```
- Query, positive documents, and hard negative documents. Here, negative examples include both the hard negatives specific to the query and random negatives from other queries' negative documents. Refer to the [example_data](https://github.com/NLPJCL/RAG-Retrieval/blob/master/example_data/t2rank_100.jsonl) file.
```
{"query": str, "pos": List[str], "neg":List[str], "prompt_for_query"(optional): str}
```
- Query and documents, along with supervised scores for each query-document pair. Refer to the [example_data](https://github.com/NLPJCL/RAG-Retrieval/blob/master/example_data/lmsft.jsonl) file. Supervised signals can be constructed in two recommended ways:
  - Manual annotation: Similar to the [STS task](https://huggingface.co/datasets/PhilipMay/stsb_multi_mt), scoring each query-document pair based on their similarity.
  - LLM annotation: Following the paper [Atlas](https://www.jmlr.org/papers/v24/23-0037.html), using the perplexity of an LLM or the FiD score of an Encoder-Decoder Transformer architecture.
```
{"query": str, "pos": List[str], "scores":List[float], "prompt_for_query"(optional): str}
```
- query and the corresponding teacher embedding. Method introduction: [infgrad/jasper_en_vision_language_v1](https://huggingface.co/infgrad/jasper_en_vision_language_v1), distilled data construction, reference: [examples/stella_embedding_distill](../../../examples/stella_embedding_distill/)
  - text: the following example.
  - teacher_embedding: a np.memmap file.
```
{"query": str, "prompt_for_query"(optional): str}
```

Note: `prompt_for_query` can be used to incorporate instructional information into the query, e.g., "Instruct: Given a user query, retrieve documents helpful for answering the query\nQuery: ".

# Training

Run the `train_embedding.sh` script to start training. Below is the code of `train_embedding.sh`.

#For BERT-like models, using fsdp (ddp)
```bash
 CUDA_VISIBLE_DEVICES="0,1"   nohup  accelerate launch \
 --config_file ../../../config/default_fsdp.yaml \
 train_embedding.py  \
 --config ./config/training_embedding.yaml  \
 >./logs/t2ranking_100_example_bert.log &
```

#For LLM-like models, using deepspeed (zero1-3)

```bash
 CUDA_VISIBLE_DEVICES="0,1"   nohup  accelerate launch \
 --config_file ../../../config/deepspeed/deepspeed_zero2.yaml \
 train_embedding.py  \
 --config ./config/training_embedding.yaml  \
 >./logs/t2ranking_100_example_llm.log &
```

#distill teacher embedding,fsdp(ddp) refer to：[infgrad/jasper_en_vision_language_v1](https://huggingface.co/infgrad/jasper_en_vision_language_v1).
```bash
 CUDA_VISIBLE_DEVICES="0,1"   nohup  accelerate launch \
 --config_file ../../../config/default_fsdp.yaml \
 train_embedding.py  \
 --config ./config/distill_embedding.yaml  \
 >./logs/t2ranking_100_example_bert_distill.log &
```

**Parameter Explanation**
- `model_name_or_path`: The name of the open-source embedding model or the local path where it has been downloaded. (As long as the model supports fine-tuning with the [sentence-transformers](https://www.sbert.net/) library, it can be used in our framework.)
- `save_on_epoch_end`: Whether to save the model at the end of each epoch.
- `log_with`: Visualization tool. If not set, default parameters will be used without errors.
- `neg_nums`: Number of hard negative examples during training. It should not exceed the number of real negative examples (`neg:List[str]`) in the training dataset. If `neg_nums` is greater than the number of real negatives, real negatives will be resampled, and `neg_nums` negatives will be randomly selected. If `neg_nums` is less than the number of real negatives, `neg_nums` negatives will be randomly chosen during training. (Ignore this parameter if only query and positive documents are available.)
- `batch_size`: Larger batch sizes result in more random negatives within the batch, improving performance. The actual number of negatives is `batch_size * neg_nums`.
- `lr`: Learning rate, typically between 1e-5 and 5e-5.
- `use_mrl` : Whether to use mlr training. (Add a new linear layer [hidden_size, max(mrl_dims)] to perform MRL training, refer to [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147), use MRL-E by default)
- `mrl_dims` : List of mrl dimensions to train, for example, "128, 256, 512, 768, 1024, 1280, 1536, 1792".

For BERT-like models, fsdp is used by default to support multi-GPU training. Here are example configuration files:
- [default_fsdp](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/default_fsdp.yaml). Use this configuration file for training a embedding model from scratch based on `chinese-roberta-wwm-ext`.
- [xlmroberta_default_config](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/xlmroberta_default_config.yaml). Use this configuration file for fine-tuning based on `bge-m3-embedding` and `bce-embedding-base_v1`, as they are trained on the multilingual `xlmroberta`.

For LLM-like models, deepspeed is used by default to support multi-GPU training. Here is an example configuration file:
- [deepspeed_zero2](https://github.com/NLPJCL/RAG-Retrieval/blob/master/config/deepspeed/deepspeed_zero2.yaml)

Modifying the multi-GPU training configuration:
- Change `CUDA_VISIBLE_DEVICES="0"` in `train_embedding.sh` to the desired multi-GPU setting.
- Modify the `num_processes` in the aforementioned configuration files to the number of GPUs you want to use.

# Loading the Model for Prediction

The saved model will be saved in the sentence-transformers format, allowing for easy loading and prediction.

In `model.py`, an example is provided on how to load and use the model for prediction.

```python
cuda_device = 'cuda:0'
ckpt_path = 'path_to_saved_model'
embedding = Embedding.from_pretrained(
    ckpt_path,
)
embedding.to(cuda_device)
input_lst = ['I love China', 'I love climbing Mount Tai']
embeddings = embedding.encode(input_lst, device=cuda_device)
```

Additionally, the `sentence-transformers` library can be used for inference.
```python
from sentence_transformers import SentenceTransformer

ckpt_path = 'path_to_saved_model'
input_lst = ['I love China', 'I love climbing Mount Tai']

model = SentenceTransformer(ckpt_path)
embeddings = model.encode(input_lst)
```

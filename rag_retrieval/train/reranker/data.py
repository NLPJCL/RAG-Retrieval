import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
import json

def map_label_to_continuous(label, min_label, max_label):
    """
    Maps a discrete label in the range [min_label, max_label] to a continuous value in [0, 1].

    Args:
        label (int): The discrete label to be mapped.
        min_label (int): The minimum value of the discrete label range.
        max_label (int): The maximum value of the discrete label range.

    Returns:
        float: A continuous value in the range [0, 1].
    """
    if label < min_label or label > max_label:
        raise ValueError("Label is out of range.")

    return (label - min_label) / (max_label - min_label)


class RankerDataset(Dataset):
    def __init__(self, train_data_path, target_model, max_len=512, max_label=1, min_label=0):
        self.model = target_model
        self.max_len = max_len
        assert max_label > min_label and min_label >= 0
        self.max_label = max_label
        self.min_label = min_label
        self.map_func = lambda x: map_label_to_continuous(x, self.min_label, self.max_label)
        self.train_data = self.read_train_data(train_data_path)

    def read_train_data(self, train_data_path):
        # standard input data type:
        # {"query": str(required), "pos": List[str](required), "neg":List[str](optional), "pos_scores": List[int](optional), "neg_scores": List[int](optional)}}   
        
        train_data = []
        with open(train_data_path) as f:
            for line in tqdm.tqdm(f):
                data_dic = json.loads(line.strip())
                assert "query" in data_dic and "pos" in data_dic
                if "pos_scores" in data_dic:
                    assert len(data_dic["pos"]) == len(data_dic["pos_scores"])
                if "neg_scores" in data_dic:
                    assert "neg" in data_dic and len(data_dic["neg"]) == len(
                        data_dic["neg_scores"]
                    )

                if "pos" in data_dic:
                    for idx, text_pos in enumerate(data_dic["pos"]):
                        pos_score = 1
                        if "pos_scores" in data_dic:
                            pos_score = self.map_func(data_dic["pos_scores"][idx])
                        train_data.append([data_dic["query"], text_pos, pos_score])
                if "neg" in data_dic:
                    for idx, text_neg in enumerate(data_dic["neg"]):
                        neg_score = 0
                        if "neg_scores" in data_dic:
                            neg_score = self.map_func(data_dic["neg_scores"][idx])
                        train_data.append([data_dic["query"], text_neg, neg_score])

        # standard output data type: [query, doc, score[0,1]]
        print(f"Loaded {len(train_data)} data")
        return train_data

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.train_data[idx]

    def collate_fn(self, batch):
        all_batch_pairs = []
        all_labels = []

        for item in batch:
            all_batch_pairs.append([item[0], item[1]])
            all_labels.append(item[2])

        tokens = self.model.preprocess(all_batch_pairs, self.max_len)
        label_batch = torch.tensor(all_labels, dtype=torch.float16)

        return tokens, label_batch


def test_RankerDataset():
    from modeling import SeqClassificationRanker

    train_data_path = "../../../example_data/t2rank_100.jsonl"
    model_name_or_path = "Qwen/Qwen2.5-1.5B"

    model = SeqClassificationRanker.from_pretrained(
        model_name_or_path=model_name_or_path,
        query_format="query: {}",
        document_format="document: {}",
    )
    dataset = RankerDataset(train_data_path, target_model=model, max_len=128)

    dataloader = DataLoader(dataset, batch_size=32, collate_fn=dataset.collate_fn)

    print(len(dataloader))

    for batch in tqdm.tqdm(dataloader):
        print(batch)
        print(model.tokenizer.batch_decode(batch[0]["input_ids"])[0])
        break


if __name__ == "__main__":
    test_RankerDataset()

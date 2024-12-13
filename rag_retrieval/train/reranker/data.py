import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
import json
from collections import defaultdict
from utils import map_label_to_continuous, visualize_label_distribution, shuffle_text


class RankerDataset(Dataset):
    def __init__(self, train_data_path, target_model, max_len=512, max_label=1, min_label=0, shuffle_rate=0.0):
        self.model = target_model
        self.max_len = max_len
        assert max_label > min_label and min_label >= 0
        self.max_label = max_label
        self.min_label = min_label
        self.map_func = lambda x: map_label_to_continuous(x, self.min_label, self.max_label)
        assert 0 <= shuffle_rate <= 1 , "shuffle rate must be between 0 and 1"
        self.shuffle_rate = shuffle_rate # The probability of shuffling the text
        
        self.train_data = self.read_train_data(train_data_path)

    def read_train_data(self, train_data_path):
        # standard input data type:
        # {"query": str(required), "pos": List[str](required), "neg":List[str](optional), "pos_scores": List(optional), "neg_scores": List(optional)}}   
        
        train_data = []
        label_distribution = defaultdict(int)
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
                        label_distribution[f"{pos_score:.2f}"] += 1
                        if self.shuffle_rate > 0:
                            text_pos = shuffle_text(text_pos, self.shuffle_rate)
                        train_data.append([data_dic["query"], text_pos, pos_score])
                if "neg" in data_dic:
                    for idx, text_neg in enumerate(data_dic["neg"]):
                        neg_score = 0
                        if "neg_scores" in data_dic:
                            neg_score = self.map_func(data_dic["neg_scores"][idx])
                        label_distribution[f"{neg_score:.2f}"] += 1
                        if self.shuffle_rate > 0:
                            text_neg = shuffle_text(text_neg, self.shuffle_rate)
                        train_data.append([data_dic["query"], text_neg, neg_score])

        # only visualize the label distribution on the main process
        if torch.distributed.get_rank() == 0:
            visualize_label_distribution(label_distribution)
            
        # standard output data type: [query, doc, score[0,1]]
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

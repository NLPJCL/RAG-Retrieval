import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
import json


class RankerDataset(Dataset):
    def __init__(
        self,
        train_data_path,
        target_model,
        max_len=512
    ):
        self.train_data = self.read_train_data(train_data_path)
        self.model = target_model
        self.max_len = max_len

    def read_train_data(self, train_data_path):
        train_data = []
        with open(train_data_path) as f:
            for line in tqdm.tqdm(f):
                data_dic = json.loads(line.strip())
                if "pos" in data_dic:
                    for text_pos in data_dic["pos"]:
                        train_data.append([data_dic["query"], text_pos, 1])
                if "neg" in data_dic:
                    for text_neg in data_dic["neg"]:
                        train_data.append([data_dic["query"], text_neg, 0])
        # 无论文件内部格式是怎么样，最终数据集标准格式均为 [query, doc, score]
        return train_data

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.train_data[idx]

    def collate_fn(self, batch):
        all_batch_pairs = []
        all_labels = []
        
        for item in batch:
            all_batch_pairs.append([item[0],item[1]])
            all_labels.append(item[2])
        
        tokens = self.model.preprocess(all_batch_pairs, self.max_len)
        label_batch = torch.tensor(all_labels, dtype=torch.float16)

        return tokens, label_batch


def test_RankerDataset():
    from modeling import SequenceClassifier
    
    train_data_path = "../../../example_data/t2rank_100.jsonl"
    model_name_or_path="Qwen/Qwen2.5-1.5B"
    
    model = SequenceClassifier.from_pretrained(
        model_name_or_path=model_name_or_path,
        query_format="query: {}",
        document_format="document: {}",
    )
    dataset = RankerDataset(
        train_data_path,
        target_model=model,
        max_len=128
    )

    dataloader = DataLoader(dataset, batch_size=32, collate_fn=dataset.collate_fn)

    print(len(dataloader))

    for batch in tqdm.tqdm(dataloader):
        print(batch)
        print(model.tokenizer.batch_decode(batch[0]["input_ids"])[0])
        break


if __name__ == "__main__":
    test_RankerDataset()

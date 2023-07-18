import os
import sys
import datasets
import logging
import json
import pandas as pd
import torch
from torchtext.vocab import GloVe, vocab
from torch.utils.data import DataLoader
from transformers import BertTokenizer

def _get_data(file_path: str):
    #HANDE: We dont have soft labels in dev anymore.
    #if 'chaosnli' in file_path and ('dev' in file_path or 'test' in file_path):
    if 'chaosnli' in file_path and 'test' in file_path:
        data = [json.loads(line) for line in open(file_path, "r")]
        dataset = pd.DataFrame(data)
        dataset["majority_label"].replace(to_replace="e", value=0, inplace=True)
        dataset["majority_label"].replace(to_replace="n", value=1, inplace=True)
        dataset["majority_label"].replace(to_replace="c", value=2, inplace=True)
        dataset["old_label"].replace(to_replace="e", value=0, inplace=True)
        dataset["old_label"].replace(to_replace="n", value=1, inplace=True)
        dataset["old_label"].replace(to_replace="c", value=2, inplace=True)
        return (
            dataset["example"].str.get("premise").tolist(),
            dataset["example"].str.get("hypothesis").tolist(),
            #dataset["majority_label"].tolist(),
            dataset["old_label"].tolist(),
        )
    else:
        data = [json.loads(line) for line in open(file_path, "r")]
        dataset = pd.DataFrame(data)
        dataset = dataset[dataset.gold_label != "-"]
        dataset["gold_label"].replace(to_replace="entailment", value=0, inplace=True)
        dataset["gold_label"].replace(to_replace="neutral", value=1, inplace=True)
        dataset["gold_label"].replace(to_replace="contradiction", value=2, inplace=True)
        return (
            dataset["sentence1"].tolist(),
            dataset["sentence2"].tolist(),
            dataset["gold_label"].tolist(),
        )


class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        # return len(self.labels)
        return len(self.encodings.input_ids)


#def collate_fn(batch):
#    return batch
#    new_batch = []
#    for i in range(len(batch)):
#        new_batch.append({'input_ids': batch[i]['input_ids'].to(device), 
#                          'token_type_ids': batch[i]['token_type_ids'].to(device),
#                          'attention_mask': batch[i]['attention_mask'].to(device)})
#    return new_batch

def preprocess_with(tokenizer):
    def preprocess(input_):
        return tokenizer(
            input_["text"],
            truncation=True,
            padding="max_length",
            max_length=50
        )
    
    return preprocess







def get_nli_dataset(config, tokenizer, data_path):

    logging.info(f"Experiment dataset: {config.dataset}")
    
    train_premises, train_hypotheses, train_labels = _get_data(
        data_path + "/" + config.dataset + "/train.jsonl"
    )
    logging.info(
        f"First training example: {train_premises[0]} --> {train_hypotheses[0]} ({train_labels[0]})"
    )

    dev_premises, dev_hypotheses, dev_labels = _get_data(
        data_path + "/" + config.dataset + "/dev.jsonl"
    )
    logging.info(
        f"First dev example: {dev_premises[0]} --> {dev_hypotheses[0]} ({dev_labels[0]})"
    )
    test_premises, test_hypotheses, test_labels = _get_data(
        data_path + "/" + config.dataset + "/test.jsonl"
    )
    logging.info(
        f"First test example: {test_premises[0]} --> {test_hypotheses[0]} ({test_labels[0]})"
    )

    train_encodings = tokenizer(
        train_premises,
        train_hypotheses,
        truncation=True,
        padding=True,
    )
    dev_encodings = tokenizer(
        dev_premises, dev_hypotheses, truncation=True, padding=True
    )
    test_encodings = tokenizer(
        test_premises,
        test_hypotheses,
        truncation=True,
        padding=True,
    )

    train_dataset = NLIDataset(train_encodings, train_labels)
    dev_dataset = NLIDataset(dev_encodings, dev_labels)
    test_dataset = NLIDataset(test_encodings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, 
                                num_workers=8, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.eval_batch_size, shuffle=False,
                                num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False,
                                num_workers=8, pin_memory=True)
    
    # We only use a subset of the data here for demonstration purposes
    #train_split = datasets.load_dataset("multi_nli", split='train')
    #test_split = datasets.load_dataset("multi_nli", split='validation_matched')
    #ood_test_split = datasets.load_dataset("multi_nli", split='validation_matched')
    #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    #train_split = train_split.rename_column("label", "labels")
    #train_split = train_split.rename_column("premise", "text")
    #train_split = train_split.map(preprocess_with(tokenizer), batched=True)
    #train_split.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    #test_split = test_split.rename_column("label", "labels")
    #test_split = test_split.rename_column("premise", "text")
    #test_split = test_split.map(preprocess_with(tokenizer), batched=True)
    #test_split.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    #ood_test_split = ood_test_split.rename_column("label", "labels")
    #ood_test_split = ood_test_split.rename_column("premise", "text")
    #ood_test_split = ood_test_split.map(preprocess_with(tokenizer), batched=True)
    #ood_test_split.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    #BATCH_SIZE = 32
    #train_loader = DataLoader(train_split, batch_size=BATCH_SIZE)
    #test_loader = DataLoader(test_split, batch_size=BATCH_SIZE)
    #ood_test_loader = DataLoader(ood_test_split, batch_size=BATCH_SIZE)

    #print('train_loader len:', len(train_loader))
    #print('test_loader len:', len(test_loader))
    #print('ood_test_loader len:', len(ood_test_loader))


    if 'LSTM' in config.model: 
        unk_token = "<unk>"
        unk_index = 0
        glove_vectors = GloVe(name='42B', dim=300)
        glove_vocab = vocab(glove_vectors.stoi)
        glove_vocab.insert_token("<unk>",unk_index)
        glove_vocab.set_default_index(unk_index)

        glove_embeddings = glove_vectors.vectors
        glove_embeddings = torch.cat((torch.zeros(1, glove_embeddings.shape[1]), glove_embeddings))

    else:
        glove_embeddings = []

    return train_loader, dev_loader, test_loader, glove_embeddings



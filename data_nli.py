import os
import sys
import datasets
import logging
import json
import pandas as pd
import torch
import jax

import jax.numpy as jnp
from flax.training.common_utils import shard, shard_prng_key

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


#TODO: use something else than torch Dataset
class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: jnp.array(val)[idx] for key, val in self.encodings.items()}
        item["labels"] = jnp.array(self.labels)[idx]
        return item

    def __len__(self):
        return len(self.encodings.input_ids)



def get_nli_datasets(config, tokenizer, data_path):

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

    return train_dataset, dev_dataset, test_dataset



#Jax dataloader:
def train_data_loader(rng, dataset, batch_size):
    steps_per_epoch = len(dataset) // batch_size
    perms = jax.random.permutation(rng, len(dataset))
    perms = perms[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    perms = perms.reshape((steps_per_epoch, batch_size))

    for perm in perms:
        batch = dataset[perm]
        batch = {k: jnp.array(v) for k, v in batch.items()}
        batch = shard(batch)

        yield batch


def eval_data_loader(dataset, batch_size):
    for i in range(len(dataset) // batch_size):
        batch = dataset[i * batch_size : (i + 1) * batch_size]
        batch = {k: jnp.array(v) for k, v in batch.items()}
        batch = shard(batch)

        yield batch




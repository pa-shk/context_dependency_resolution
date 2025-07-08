import json
from pathlib import Path
from typing import Callable

import pandas as pd
from datasets import Dataset, DatasetDict


def load_ds(dataset_name: str) -> DatasetDict:
    """
    Loads and splits a dataset from a JSON file into train/val/test sets.

    Args:
        dataset_name: Filename of the JSON dataset in the data directory
        
    Returns:
        DatasetDict with keys: 'train', 'val', 'test'
    """
    data_path = Path.cwd().parent.parent.parent / "data" / dataset_name
    with open(data_path) as f:
        data = json.load(f)

    history, phrase, rewrite = [], [], []
    for sample in data:
        history.append(sample["History"])
        phrase.append(sample["Phrase"])
        rewrite.append(sample["Rewrite"])

    df = pd.DataFrame({"history": history, "phrase": phrase, "rewrite": rewrite})
    ds = Dataset.from_pandas(df)

    ds = ds.train_test_split(test_size=0.2, shuffle=True, seed=42)
    train_ds, test_ds = ds["train"], ds["test"]
    test_ds = test_ds.train_test_split(test_size=0.5, shuffle=True, seed=42)
    val_ds, test_ds = test_ds["train"], test_ds["test"]

    ds["train"], ds["val"], ds["test"] = train_ds, val_ds, test_ds
    return ds

def tokenize_ds(ds: DatasetDict, preprocess_func: Callable) -> DatasetDict:
    """
    Tokenizes a DatasetDict using a preprocessing function.
    
    Args:
        ds: Dataset dictionary to tokenize
        preprocess_func: Function that takes a batch of data and returns
                         tokenized results. Should handle all required
                         columns in the dataset.
    
    Returns:
        Tokenized DatasetDict with the same structure as input
    """
    return ds.map(preprocess_func, batched=True, batch_size=4)

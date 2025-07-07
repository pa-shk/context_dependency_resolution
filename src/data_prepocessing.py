import json
from datasets import Dataset
import pandas as pd
from pathlib import Path


def load_ds(dataset_name):
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
    val_ds, test_ds = test_ds["train"], test_ds["train"]

    ds["train"] = train_ds
    ds["val"] = val_ds
    ds["test"] = test_ds
    
    return ds

def tokenize_ds(ds, preprocess_func):
    return ds.map(preprocess_func, batched=True, batch_size=4)
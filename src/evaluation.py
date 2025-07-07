import sys
sys.path.append('../../..')

import evaluate
import pandas as pd
import torch
from rouge_metric import PyRouge

from src.metrics import RestorationFScore


bleu = evaluate.load("sacrebleu")
rouge = PyRouge(rouge_n=(4), skip_gap=4)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluator:
    def __init__(self, dataset, model, tokenizer, infer_func, client=None):
        self.dataset = dataset
        self.test_results = pd.DataFrame()
        self.model = model
        self.tokenizer = tokenizer
        self.infer_func = infer_func
        self.client = client

    def evaluate(self):
        self.test_results = pd.DataFrame(
            self.infer_func(dataset=self.dataset, 
                            model=self.model, 
                            client=self.client), 
            columns=["model_out_raw"]
        )

        self.test_results["history"] = self.dataset["test"]["history"]
        self.test_results["text"] = self.dataset["test"]["phrase"]
        self.test_results["restored_text"] = self.dataset["test"]["rewrite"]
        self.test_results.to_csv(f"test_results.csv")
        
        self.test_results["type"] = "2rca"
        self.test_results.groupby(by="type").agg(list)
        return self.test_results.apply(self.callculate_metrics, axis=1).drop(
            columns=["model_out_raw", "history", "text", "restored_text"]
        )
    
    def callculate_metrics(self, row):
        row["bleu_score"] = bleu.compute(predictions=row.model_out_raw,
                                         references=row.text)["score"]

        rouge_scores = rouge.evaluate(row.model_out_raw,
                                      [[t] for t in row.text])
        for k in rouge_scores:
            row[k] = rouge_scores[k]["f"]

        for n in range(1, 5):
            rf_score = RestorationFScore(self.tokenizer, n)
            row[f"rf_score_{n}"] = rf_score.evaluate(predictions=row.model_out_raw,
                                                     references=row.text,
                                                     rewrites=row.restored_text)
        return row

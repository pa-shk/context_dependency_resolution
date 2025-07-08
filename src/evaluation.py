import sys

sys.path.append('../../..')

from typing import Any, Callable, List, Optional, Union

import evaluate
import pandas as pd
import torch
from datasets import DatasetDict
from rouge_metric import PyRouge
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from src.metrics import RestorationFScore

bleu = evaluate.load("sacrebleu")
rouge = PyRouge(rouge_n=(4), skip_gap=4)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Evaluator:
    """Evaluator class for model performance assessment.
    
    Calculates BLEU, ROUGE, and Restoration F-Scores for model outputs.
    
    Attributes:
        dataset: HF Dataset containing test data
        test_results: DataFrame storing evaluation results
        model: Model to be evaluated
        tokenizer: Tokenizer for text processing
        infer_func: Inference function for model predictions
        client: Optional client for model accessed via API
    """

    def __init__(
        self,
        dataset: DatasetDict,
        model: torch.nn.Module,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        infer_func: Callable[..., List[str]],
        client: Optional[Any] = None
    ) -> None:
        """
        Initialize Evaluator instance.
        
        Args:
            dataset: HF dataset
            model: Trained model for evaluation
            tokenizer: Tokenizer compatible with the model
            infer_func: Function that takes (dataset, model, client) and
                        returns list of raw model outputs
            client: Optional client object for distributed systems
        """
        self.dataset = dataset
        self.test_results = pd.DataFrame()
        self.model = model
        self.tokenizer = tokenizer
        self.infer_func = infer_func
        self.client = client

    def evaluate(self) -> pd.DataFrame:
        """
        Run full evaluation pipeline.
        
        Returns:
            DataFrame with calculated metrics
        """
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
        return self.test_results.apply(self.calculate_metrics, axis=1).drop(
            columns=["model_out_raw", "history", "text", "restored_text"]
        )
    
    def calculate_metrics(self, row: pd.Series) -> pd.Series:
        """
        Calculate metrics for a single data row.
        
        Args:
            row: DataFrame row containing list of model output and ground truths
            
        Returns:
            Input row with calculated metrics
        """
        row["bleu_score"] = bleu.compute(
            predictions=[row.model_out_raw],
            references=[[row.text]]
        )["score"]

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

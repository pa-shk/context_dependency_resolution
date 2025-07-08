from typing import Iterable, List, Set, Tuple, Union

import evaluate
import numpy as np
from rouge_metric import PyRouge
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

bleu = evaluate.load("sacrebleu")
rouge = PyRouge(rouge_n=(4), skip_gap=4)


class RestorationFScore:
    """Computes Restoration F-Score for text restoration tasks.
    
    Args:
        tokenizer: Tokenizer for text processing
        n_gram: N-gram size to use for comparison (default=2)
    """

    def __init__(self, 
                 tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], 
                 n_gram: int = 2) -> None:
        self.n_gram = n_gram
        self.tokenizer = tokenizer

    def preprocess(self, sents: List[str]) -> Iterable[Set[Tuple[int, ...]]]:
        """
        Tokenizes sentences and converts to n-gram sets.
        
        Args:
            sents: List of sentences to process
            
        Yields:
            Set of n-gram tuples for each sentence
        """
        for sent in sents:
            sent_tokenize = self.tokenizer(sent)['input_ids']
            yield [tuple(sent_tokenize[i:i+self.n_gram]) for i, _ in enumerate(sent_tokenize)]

    def _iterate_eval(self) -> Iterable[float]:
        """
        Computes F-scores for each sample in the dataset.
        
        Yields:
            F-score for each sample
        """
        for i, predictions in enumerate(self.predictions):
            restored_ngrams = set(predictions).difference(self.references[i])
            ngrams_in_ref = set(self.rewrites[i]).difference(self.references[i])
            interagree = ngrams_in_ref.intersection(restored_ngrams)
            if len(restored_ngrams):
                precision = len(interagree) / len(restored_ngrams)
            else:
                precision = 0.
            if len(ngrams_in_ref):
                recall = len(interagree) / len(ngrams_in_ref)
            else:
                recall = 0.
            if precision or recall:
                yield 2 * ((precision * recall) / (precision + recall))
            else:
                yield 0.

    def evaluate(self, 
                predictions: List[str], 
                references: List[str], 
                rewrites: List[str]) -> float:
        """
        Computes average Restoration F-Score across all samples.
        
        Args:
            predictions: Model outputs
            references: Original references
            rewrites: Ground truth restored text
            
        Returns:
            Mean F-score across all samples
        """
        self.predictions = [p for p in self.preprocess(predictions)]
        self.references = [p for p in self.preprocess(references)]
        self.rewrites = [p for p in self.preprocess(rewrites)]
        return np.mean(list(self._itereval()))

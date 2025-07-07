import numpy as np
import evaluate
from rouge_metric import PyRouge

bleu = evaluate.load("sacrebleu")
rouge = PyRouge(rouge_n=(4), skip_gap=4)


class RestorationFScore:

    def __init__(self, tokenizer, n_gram: int=2):
        self.n_gram = n_gram
        self.tokenizer = tokenizer

    def preprocess(self, sents):
        for sent in sents:
            sent_tokenize = self.tokenizer(sent)['input_ids']
            yield [tuple(sent_tokenize[i:i+self.n_gram]) for i, _ in enumerate(sent_tokenize)]

    def _itereval(self):
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

    def evaluate(self, predictions: list,
                 references: list, rewrites: list):
        self.predictions = [p for p in self.preprocess(predictions)]
        self.references = [p for p in self.preprocess(references)]
        self.rewrites = [p for p in self.preprocess(rewrites)]
        return np.mean(list(self._itereval()))


def callculate_metrics(row):
    row["bleu_score"] = bleu.compute(predictions=row.model_out_raw,
                                     references=row.text)['score']

    rouge_scores = rouge.evaluate(row.model_out_raw,
                                  [[t] for t in row.text])
    for k in rouge_scores:
        row[k] = rouge_scores[k]['f']

    for n in range(1, 5):
        rf_score = RestorationFScore(tokenizer, n)
        row[f"rf_score_{n}"] = rf_score.evaluate(predictions=row.model_out_raw,
                                                 references=row.text,
                                                 rewrites=row.restored_text)
    return row
from collections import Counter
from pathlib import Path
import re
import unicodedata

from bert_score import BERTScorer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import pandas as pd


class BERTScore:
    def __init__(self, lang):
        assert lang in ["English", "Japanese"]
        self.scorer = BERTScorer(lang="en" if lang == "English" else "ja")
        self.results = []

    def compute(self, reference, candidate):
        precision, recall, f1 = self.scorer.score([reference], [candidate])
        self.results.append([precision.item(), recall.item(), f1.item()])

    def average(self, column):
        assert column in ["P", "R", "F1"]
        if not self.results:
            return 0.0
        col_index = {"P": 0, "R": 1, "F1": 2}[column]
        return round(sum(row[col_index] for row in self.results) / len(self.results), 6)

    def save(self, save_path):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.results, columns=["P", "R", "F1"]).to_csv(save_path, index=False)
        print(f"BERTScore saved to {save_path}.")


class BLEUScore:
    def __init__(self, weights=(0.25, 0.25, 0, 0)):
        self.weights = weights
        self.results = []
        self.smooth = SmoothingFunction().method1

    def compute(self, reference, candidate):
        if not reference.strip() or not candidate.strip():
            self.results.append(0.0)
            return
        score = sentence_bleu(
            [reference.split()],
            candidate.split(),
            weights=self.weights,
            smoothing_function=self.smooth,
        )
        self.results.append(score)

    def average(self):
        if not self.results:
            return 0.0
        return round(sum(self.results) / len(self.results), 6)

    def save(self, save_path):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.results, columns=["BLEU"]).to_csv(save_path, index=False)
        print(f"BLEU scores saved to {save_path}.")


class Statistic:
    def __init__(self, lang, interjection_patterns):
        assert lang in ("English", "Japanese")
        self.lang = lang
        self.counter = Counter()
        self.total_token_count = 0

        normalize = lambda s: unicodedata.normalize("NFKC", s).lower()
        patterns = sorted({normalize(pattern) for pattern in interjection_patterns}, key=lambda item: -len(item))
        alt = "|".join(re.escape(pattern) for pattern in patterns)
        self.regex = re.compile(rf"(?i)(?<!\w)({alt})(?!\w)")

    def _tokenize(self, text):
        normalized = unicodedata.normalize("NFKC", text).lower()
        if self.lang == "English":
            return re.findall(r"\w+", normalized)
        return list(normalized)

    def compute(self, text):
        normalized = unicodedata.normalize("NFKC", text)
        self.total_token_count += len(self._tokenize(normalized))
        for match in self.regex.finditer(normalized):
            self.counter[match.group(1).lower()] += 1

    def average(self):
        if self.total_token_count == 0:
            return 0.0
        return round(sum(self.counter.values()) / self.total_token_count, 6)

    def types(self):
        return len(self.counter)

    def save(self, save_path):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for pattern, count in self.counter.items():
            frequency = count / self.total_token_count if self.total_token_count else 0.0
            rows.append({"pattern": pattern, "count": count, "frequency": round(frequency, 6)})
        pd.DataFrame(rows).to_csv(save_path, index=False)
        print(f"Interjection stats saved to {save_path}")

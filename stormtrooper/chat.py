import os
import warnings
from abc import ABC, abstractmethod
from typing import Iterable, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from thefuzz import process
from tqdm import tqdm

default_system_prompt = """
You are a classification model that is really good at following
instructions and produces brief answers
that users can use as data right away.
Please follow the user's instructions as precisely as you can.
"""

default_prompt = """
Your task will be to classify a text document into one
of the following classes: {classes}.
Please respond with a single label that you think fits
the document best.
{examples}
Classify the following piece of text:
 ```{X}```
"""

example_prompt = """
Examples of texts labelled '{label}':
{examples}
"""


class ChatClassifier(BaseEstimator, ClassifierMixin, ABC):
    prompt = default_prompt
    system_prompt = default_system_prompt
    progress_bar = True
    fuzzy_match = True

    def fit(self, X: Optional[Iterable[str]], y: Iterable[str]):
        """Learns class labels.

        Parameters
        ----------
        X: iterable of str or None
            Examples to pass into the few-shot prompt.
            If None, the model is zero-shot.
        y: iterable of str
            Iterable of class labels.
            Should at least contain a representative sample
            of potential labels.

        Returns
        -------
        self
            Fitted model.
        """
        if X is not None:
            self.examples_ = dict()
            for text, label in zip(X, y):
                if label not in self.examples_:
                    self.examples_[label] = []
                self.examples_[label].append(text)
            self.classes_ = np.array(list(self.examples_.keys()))
        else:
            self.classes_ = np.array(list(set(y)))
        self.n_classes = len(self.classes_)
        return self

    def get_user_prompt(self, text: str) -> str:
        if getattr(self, "classes_", None) is None:
            raise NotFittedError("No class labels have been learnt yet, fit the model.")
        if getattr(self, "examples_", None) is not None:
            text_examples = []
            for label, examples in self.examples_.items():
                examples = [f"'{example}'" for example in examples]
                subprompt = example_prompt.format(
                    label=label, examples="\n".join(examples)
                )
                text_examples.append(subprompt)
            examples_subprompt = "\n".join(text_examples)
        else:
            examples_subprompt = ""
        classes_in_quotes = [f"'{label}'" for label in self.classes_]
        prompt = self.prompt.format(
            X=text,
            classes=", ".join(classes_in_quotes),
            examples=examples_subprompt,
        )
        return prompt

    def generate_messages(self, text: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.get_user_prompt(text)},
        ]

    def partial_fit(self, X: Optional[Iterable[str]], y: Iterable[str]):
        """Learns class labels.
        Can learn new labels if new are encountered in the data.

        Parameters
        ----------
        X: iterable of str or None
            Examples to pass into the few-shot prompt.
        y: iterable of str
            Iterable of class labels.

        Returns
        -------
        self
            Fitted model.
        """
        if X is None:
            self.classes_ = np.array(set(self.classes_) | set(y))
        else:
            for text, label in zip(X, y):
                if label not in self.examples_:
                    self.examples_[label] = []
                self.examples_[label].append(text)
            self.classes_ = np.array(list(self.examples_.keys()))
        self.n_classes = len(self.classes_)
        return self

    @abstractmethod
    def predict_one(self, text: str) -> str:
        pass

    def predict(self, X: Iterable[str]) -> np.ndarray:
        """Predicts most probable class label for given texts.

        Parameters
        ----------
        X: iterable of str
            Texts to label.

        Returns
        -------
        array of shape (n_texts)
            Array of string class labels.
        """
        if self.classes_ is None:
            raise NotFittedError(
                "Class labels have not been collected yet, please fit."
            )
        pred = []
        if self.progress_bar:
            X = tqdm(X)
        for text in X:
            label = self.predict_one(text)
            if self.fuzzy_match and label not in self.classes_:
                label, _ = process.extractOne(label, self.classes_)
            pred.append(label)
        return np.array(pred)

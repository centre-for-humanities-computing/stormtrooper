from typing import Iterable, Optional

import datasets
import numpy as np
from setfit import (SetFitModel, Trainer, TrainingArguments,
                    get_templated_dataset)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError


class SetFitClassifier(BaseEstimator, ClassifierMixin):
    """"""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        sample_size: int = 8,
        device: str = "cpu",
        n_epochs: int = 10,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.classes_ = None
        self.model = SetFitModel.from_pretrained(model_name)
        self.model.model_body.to(device)
        self.trainer = None
        self.sample_size = sample_size
        self.device = device
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def _get_train_dataset(self) -> datasets.Dataset:
        if getattr(self, "examples_", None) is not None:
            X = []
            y = []
            for label, xs in self.examples_.items():
                X.extend(xs)
                y.extend([label] * len(xs))
            return datasets.Dataset.from_dict(dict(text=X, label=y))
        else:
            return get_templated_dataset(
                candidate_labels=list(self.classes_), sample_size=self.sample_size
            )

    def fit(self, X: Optional[Iterable[str]], y: Iterable[str]):
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
        self.trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                num_epochs=self.n_epochs, batch_size=self.batch_size
            ),
            train_dataset=self._get_train_dataset(),
        )
        self.trainer.train()
        return self

    def predict(self, X: Iterable[str]) -> np.ndarray:
        X = list(X)
        if getattr(self, "classes_", None) is None:
            raise NotFittedError("You need to fit the model before running inference.")
        return self.model(X)

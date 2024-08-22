from typing import Iterable, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from stormtrooper._setfit import generate_synthetic_samples, finetune_contrastive


class SetFitClassifier(BaseEstimator, ClassifierMixin):
    """Zero and few-shot classifier using the SetFit technique with encoder models.

    Parameters
    ----------
    model_name: str, default 'sentence-transformers/all-MiniLM-L6-v2'
        Name of the encoder model.
    device: str, default 'cpu'
        Device to train and run the model on.
    classification_head: ClassifierMixin, default None
        Classifier to use as the last step.
        Defaults to Logistic Regression when not specified.
    n_epochs: int, default 10
        Number of trainig epochs.
    batch_size: int, default 8
        Batch size to use during training.
    sample_size: int, default 8
        Number of  training samples to generate (only zero-shot)
    template_sentence: str, default "This sentence is {label}"
        Template sentence for synthetic samples (only zero-shot)
    random_state: int, default 42
        Seed to use for stochastic training.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        classification_head: Optional[ClassifierMixin] = None,
        device: str = "cpu",
        n_epochs: int = 10,
        batch_size: int = 32,
        sample_size: int = 8,
        template_sentence: str = "This sentence is {label}.",
        random_state: int = 42,
    ):
        self.model_name = model_name
        self.classes_ = None
        self.template_sentence = template_sentence
        self.random_state = random_state
        self.encoder = SentenceTransformer(model_name, device=device)
        if classification_head is None:
            self.classification_head = LogisticRegression()
        else:
            self.classification_head = classification_head
        self.trainer = None
        self.sample_size = sample_size
        self.device = device
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def fit(self, X: Optional[Iterable[str]], y: Iterable[str]):
        if X is not None:
            self.examples_ = dict()
            for text, label in zip(X, y):
                if label not in self.examples_:
                    self.examples_[label] = []
                self.examples_[label].append(text)
        if X is None:
            X, y = generate_synthetic_samples(
                y,
                n_sample_per_label=self.sample_size,
                template_sentence=self.template_sentence,
            )
        self.encoder = finetune_contrastive(
            self.encoder,
            X,
            y,
            n_epochs=self.n_epochs,
            seed=self.random_state,
        )
        X = list(X)
        X_embeddings = self.encoder.encode(X)
        self.classification_head.fit(X_embeddings, y)
        self.classes_ = self.classification_head.classes_
        self.n_classes = len(self.classes_)
        return self

    def predict(self, X: Iterable[str]) -> np.ndarray:
        if getattr(self, "classes_", None) is None:
            raise NotFittedError("You need to fit the model before running inference.")
        return self.classification_head.predict(self.encoder.encode(X))

    def predict_proba(self, X: Iterable[str]) -> np.ndarray:
        if getattr(self, "classes_", None) is None:
            raise NotFittedError("You need to fit the model before running inference.")
        return self.classification_head.predict_proba(self.encoder.encode(X))

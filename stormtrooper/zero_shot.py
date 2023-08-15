"""Contains zero shot classification model."""
from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.exceptions import NotFittedError
from tqdm import tqdm
from transformers import pipeline

__all__ = ["ZeroShotClassifier"]


class ZeroShotClassifier(BaseEstimator, TransformerMixin, ClassifierMixin):
    """Scikit-learn compatible zero shot classification
    with HuggingFace Transformers.

    Parameters
    ----------
    model_name: str, default 'facebook/bart-large-mnli'
        Zero-shot model to load from HuggingFace.
    progress_bar: bool, default True
        Indicates whether a progress bar should be shown.
    device: str, default 'cpu'
        Indicates which device should be used for classification.
        Models are by default run on CPU.

    Attributes
    ----------
    classes_: array of str
        Class names learned from the labels.
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        progress_bar: bool = True,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.pipe = pipeline(model=model_name, device=device)
        self.classes_ = None
        self.pandas_out = False
        self.progress_bar = progress_bar

    def fit(self, X, y: Iterable[str]):
        """Learns class labels.

        Parameters
        ----------
        X: Any
            Ignored
        y: iterable of str
            Iterable of class labels.
            Should at least contain a representative sample
            of potential labels.

        Returns
        -------
        self
            Fitted model.
        """
        self.classes_ = np.array(list(set(y)))
        self.n_classes = len(self.classes_)
        return self

    def partial_fit(self, X, y: Iterable[str]):
        """Learns class labels.
        Can learn new labels if new are encountered in the data.

        Parameters
        ----------
        X: Any
            Ignored
        y: iterable of str
            Iterable of class labels.

        Returns
        -------
        self
            Fitted model.
        """
        if self.classes_ is None:
            self.classes_ = np.array(list(set(y)))
        else:
            new_labels = set(self.classes_) - set(y)
            if new_labels:
                self.classes_ = np.concatenate(self.classes_, list(new_labels))
        self.n_classes = len(self.classes_)
        return self

    def predict_proba(self, X: Iterable[str]) -> np.ndarray:
        """Predicts class probabilities for given texts.

        Parameters
        ----------
        X: iterable of str
            Texts to predict probabilities for.

        Returns
        -------
        array of shape (n_texts, n_classes)
            Class probabilities for each text.
        """
        if self.classes_ is None:
            raise NotFittedError(
                "No class labels have been learned by the model, please fit()."
            )
        X = list(X)
        n_texts = len(X)
        res = np.empty((n_texts, self.n_classes))
        if self.progress_bar:
            X = tqdm(X)
        for i_doc, text in enumerate(X):
            out = self.pipe(text, candidate_labels=self.classes_)
            label_to_score = dict(zip(out["labels"], out["scores"]))  # type: ignore
            for i_class, label in enumerate(self.classes_):
                res[i_doc, i_class] = label_to_score[label]
        return res

    def transform(self, X: Iterable[str]):
        """Predicts class probabilities for given texts.

        Parameters
        ----------
        X: iterable of str
            Texts to predict probabilities for.

        Returns
        -------
        array of shape (n_texts, n_classes)
            Class probabilities for each text.
        """
        res = self.predict_proba(X)
        if self.pandas_out:
            import pandas as pd

            return pd.DataFrame(res, columns=self.classes_)
        else:
            return res

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
        probs = self.transform(X)
        label_indices = np.argmax(probs, axis=1)
        return self.classes_[label_indices]  # type: ignore

    def get_feature_names_out(self) -> np.ndarray:
        if self.classes_ is None:
            raise NotFittedError(
                "No class labels have been learned by the model, please fit()."
            )
        return self.classes_

    @property
    def class_to_index(self) -> dict[str, int]:
        if self.classes_ is None:
            raise NotFittedError(
                "No class labels have been learned by the model, please fit()."
            )
        return dict(zip(self.classes_, range(self.n_classes)))

    def set_output(self, transform=None):
        """Set output of the transform() function to be a dataframe instead of
        a matrix if you pass transform='pandas'.
        Otherwise it will disable pandas output."""
        if transform == "pandas":
            self.pandas_out = True
        else:
            self.pandas_out = False
        return self

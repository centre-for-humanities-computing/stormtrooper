"""Zero and few-shot classification using SetFit."""
from typing import Iterable

import datasets
import numpy as np
from setfit import SetFitModel, SetFitTrainer, get_templated_dataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

__all__ = ["SetFitZeroShotClassifier", "SetFitFewShotClassifier"]


class SetFitZeroShotClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible zero shot classification
    with SetFit and sentence transformers.

    Parameters
    ----------
    model_name: str, default 'sentence-transformers/all-MiniLM-L6-v2'
        Name of sentence transformer on HuggingFace Hub.
    sample_size: int, default 8
        Number of training samples to generate.
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
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        sample_size: int = 8,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.classes_ = None
        self.model = SetFitModel.from_pretrained(model_name)
        self.model.model_body.to(device)
        self.trainer = None
        self.sample_size = sample_size
        self.device = device

    def _train(self):
        if self.classes_ is None:
            raise NotFittedError(
                "You should collect all class labels before calling _train()"
            )
        train_dataset = get_templated_dataset(
            candidate_labels=list(self.classes_), sample_size=self.sample_size
        )
        self.trainer = SetFitTrainer(
            model=self.model, train_dataset=train_dataset
        )
        self.trainer.train()

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
        self._train()
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
            self._train()
        else:
            new_labels = set(self.classes_) - set(y)
            if new_labels:
                self.classes_ = np.concatenate(self.classes_, list(new_labels))
                self._train()
        self.n_classes = len(self.classes_)
        return self

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
        X = list(X)
        if self.classes_ is None:
            raise NotFittedError(
                "Class labels have not been collected yet, please fit."
            )
        pred_ind = self.model(X)
        pred = self.classes_[pred_ind]
        # Produces a single str when one example is passed
        # and we do not like that.
        if isinstance(pred, str):
            return np.array([pred])
        return pred


class SetFitFewShotClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible few shot classification
    with SetFit and sentence transformers.

    Parameters
    ----------
    model_name: str, default 'sentence-transformers/all-MiniLM-L6-v2'
        Name of sentence transformer on HuggingFace Hub.
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
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.classes_ = None
        self.model = SetFitModel.from_pretrained(model_name)
        self.model.model_body.to(device)
        self.trainer = None
        self.device = device

    def fit(self, X: Iterable[str], y: Iterable[str]):
        """Learns class labels.

        Parameters
        ----------
        X: iterable of str
            Examples to pass into the few-shot prompt.
        y: iterable of str
            Iterable of class labels.

        Returns
        -------
        self
            Fitted model.
        """

        X = list(X)
        y = list(y)
        self.classes_ = np.array(list(set(y)))
        self.n_classes = len(self.classes_)
        train_dataset = datasets.Dataset.from_dict(dict(text=X, label=y))
        self.trainer = SetFitTrainer(
            model=self.model, train_dataset=train_dataset
        )
        self.trainer.train()
        return self

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
        X = list(X)
        if self.trainer is None:
            raise NotFittedError("You need to fit the model before inference.")
        return self.model(X)

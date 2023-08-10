"""Zero shot classification with text-to-text language models."""
from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.exceptions import NotFittedError
from thefuzz import process
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

default_prompt = """
I will give you a piece of text. Please classify it
as one of these classes: {classes}. Please only respond
with the class label in the same format as provided.

{X}
"""


class TextToTextZeroShotClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible zero shot classification
    with generative language models.

    Parameters
    ----------
    model_name: str, default 'google/flan-t5-base'
        Text-to-text instruct model from HuggingFace.
    prompt: str, optional
        You can specify the prompt which will be used to prompt the model.
        Use placeholders to indicate where the class labels and the
        data should be placed in the prompt.
        example: '''
        I will give you a piece of text. Please classify it
        as one of these classes: {classes}. Please only respond
        with the class label in the same format as provided.

        {X}
        '''
    max_new_tokens: int, default 256
        Maximum number of tokens the model should generate.
    fuzzy_match: bool, default True
        Indicates whether the output lables should be fuzzy matched
        to the learnt class labels.
        This is useful when the model isn't giving specific enough answers.
    progress_bar: bool, default True
        Indicates whether a progress bar should be shown.

    Attributes
    ----------
    classes_: array of str
        Class names learned from the labels.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        prompt: str = default_prompt,
        max_new_tokens: int = 256,
        fuzzy_match: bool = True,
        progress_bar: bool = True,
    ):
        self.model_name = model_name
        self.prompt = prompt
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.classes_ = None
        self.progress_bar = progress_bar
        self.max_new_tokens = max_new_tokens
        self.fuzzy_match = fuzzy_match

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
            inputs = self.tokenizer(
                self.prompt.format(X=text, classes=", ".join(self.classes_)),
                return_tensors="pt",
            )
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )
            label = self.tokenizer.decode(output[0], skip_special_tokens=True)
            if self.fuzzy_match and label not in self.classes_:
                label, _ = process.extractOne(label, self.classes_)
            pred.append(label)
        return np.array(pred)

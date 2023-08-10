"""Zero shot classification with generative language models."""
from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

default_prompt = """
### System:
    You are a classification model that is really good at following
    instructions and produces brief answers
    that users can use as data right away.
    Please follow the user's instructions as precisely as you can.
### User:
    I will now give you a document that contains text
    that you will have to classify:
    '{X}'
    This text could belong to one of the following classes:
    {classes}
    Please respond with a single label that you think fits
    the document best.
### Assistant:
"""


class GenerativeZeroShotClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible zero shot classification
    with generative language models.

    Parameters
    ----------
    model_name: str, default 'upstage/Llama-2-70b-instruct-v2 '
        Generative instruct model from HuggingFace.
    prompt: str, optional
        You can specify the prompt which will be used to prompt the model.
        Use placeholders to indicate where the class labels and the
        data should be placed in the prompt.
        example: '''
        ### System:
            You are an expert of literary analysis. Help the user by following
            instructions exactly.
        ### User:
            I will give you a piece of text that belongs to
            the following classes: {classes}.
            Please respond with the topic you think this document belongs to.
            Remember to only give a label and nothing else.
            {X}.
        ### Assistant:
        '''
    progress_bar: bool, default True
        Indicates whether a progress bar should be shown.

    Attributes
    ----------
    classes_: array of str
        Class names learned from the labels.
    """

    def __init__(
        self,
        model_name: str = "upstage/Llama-2-70b-instruct-v2",
        prompt: str = default_prompt,
        progress_bar: bool = True,
    ):
        self.model_name = model_name
        self.prompt = prompt
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
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
            output = self.model.generate(**inputs, max_new_tokens=50)
            pred.append(
                self.tokenizer.decode(output[0], skip_special_tokens=True)
            )
        return np.array(pred)

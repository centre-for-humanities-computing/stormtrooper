"""Zero shot classification with text-to-text language models."""
from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from thefuzz import process
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

__all__ = ["Text2TextZeroShotClassifier", "Text2TextFewShotClassifier"]

default_prompt = """
I will give you a piece of text. Please classify it
as one of these classes: {classes}. Please only respond
with the class label in the same format as provided.

'{X}'
"""


class Text2TextZeroShotClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible zero shot classification
    with seq2seq language models.

    Parameters
    ----------
    model_name: str, default 'google/flan-t5-base'
        Text-to-text instruct model from HuggingFace.
    prompt: str, optional
        You can specify the prompt which will be used to prompt the model.
        Use placeholders to indicate where the class labels and the
        data should be placed in the prompt.
    max_new_tokens: int, default 256
        Maximum number of tokens the model should generate.
    fuzzy_match: bool, default True
        Indicates whether the output lables should be fuzzy matched
        to the learnt class labels.
        This is useful when the model isn't giving specific enough answers.
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
        model_name: str = "google/flan-t5-base",
        prompt: str = default_prompt,
        max_new_tokens: int = 256,
        fuzzy_match: bool = True,
        progress_bar: bool = True,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.prompt = prompt
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(
            self.device
        )
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
            classes_in_quotes = [f"'{label}'" for label in self.classes_]
            inputs = self.tokenizer(
                self.prompt.format(
                    X=text, classes=", ".join(classes_in_quotes)
                ),
                return_tensors="pt",
            ).to(self.device)
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )
            label = self.tokenizer.decode(output[0], skip_special_tokens=True)
            if self.fuzzy_match and label not in self.classes_:
                label, _ = process.extractOne(label, self.classes_)
            pred.append(label)
        return np.array(pred)


fewshot_prompt = """
I will give you a piece of text. Please classify it
as one of these classes: {classes}. Please only respond
with the class label in the same format as provided.
Here are some examples of texts labelled by experts.
{examples}
Label this piece of text:
'{X}'
"""

example_prompt = """
Examples of texts labelled '{label}':
{examples}
"""


class Text2TextFewShotClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible few-shot classification
    with seq2seq language models.

    Parameters
    ----------
    model_name: str, default 'google/flan-t5-base'
        Text-to-text instruct model from HuggingFace.
    prompt: str, optional
        You can specify the prompt which will be used to prompt the model.
        Use placeholders to indicate where the class labels and the
        data should be placed in the prompt.
    max_new_tokens: int, default 256
        Maximum number of tokens the model should generate.
    fuzzy_match: bool, default True
        Indicates whether the output lables should be fuzzy matched
        to the learnt class labels.
        This is useful when the model isn't giving specific enough answers.
    progress_bar: bool, default True
        Indicates whether a progress bar should be shown.
    device: str, default 'cpu'
        Indicates which device should be used for classification.
        Models are by default run on CPU.

    Attributes
    ----------
    classes_: array of str
        Class names learned from the labels.
    examples_: dict of str to list of str
        Learned examples for each class.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        prompt: str = default_prompt,
        max_new_tokens: int = 256,
        fuzzy_match: bool = True,
        progress_bar: bool = True,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.prompt = prompt
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(
            self.device
        )
        self.classes_ = None
        self.progress_bar = progress_bar
        self.max_new_tokens = max_new_tokens
        self.fuzzy_match = fuzzy_match
        self.examples_: dict[str, list[str]] = dict()

    def fit(self, X: Iterable[str], y: Iterable[str]):
        """Learns class labels.

        Parameters
        ----------
        X: iterable of str
            Examples to pass into the few-shot prompt.
        y: iterable of str
            Iterable of class labels.
            Should at least contain a representative sample
            of potential labels.

        Returns
        -------
        self
            Fitted model.
        """
        self.examples_ = dict()
        for text, label in zip(X, y):
            if label not in self.examples_:
                self.examples_[label] = []
            self.examples_[label].append(text)
        self.classes_ = np.array(list(self.examples_.keys()))
        self.n_classes = len(self.classes_)
        return self

    def partial_fit(self, X: Iterable[str], y: Iterable[str]):
        """Learns class labels.
        Can learn new labels if new are encountered in the data.

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
        for text, label in zip(X, y):
            if label not in self.examples_:
                self.examples_[label] = []
            self.examples_[label].append(text)
        self.classes_ = np.array(list(self.examples_.keys()))
        self.n_classes = len(self.classes_)
        return self

    def generate_prompt(self, text: str) -> str:
        """Generates prompt based on the model's parameters."""
        if self.examples_ is None:
            raise NotFittedError(
                "No examples have been learnt yet, fit the model."
            )
        text_examples = []
        for label, examples in self.examples_.items():
            examples = [f"'{example}'" for example in examples]
            subprompt = example_prompt.format(
                label=f"'{label}'", examples="\n".join(examples)
            )
            text_examples.append(subprompt)
        examples_subprompt = "\n".join(text_examples)
        classes_in_quotes = [f"'{label}'" for label in self.classes_]
        prompt = fewshot_prompt.format(
            X=text,
            classes=", ".join(classes_in_quotes),
            examples=examples_subprompt,
        )
        return prompt

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
            prompt = self.generate_prompt(text)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
            ).to(self.device)
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )
            label = self.tokenizer.decode(output[0], skip_special_tokens=True)
            if self.fuzzy_match and label not in self.classes_:
                label, _ = process.extractOne(label, self.classes_)
            pred.append(label)
        return np.array(pred)

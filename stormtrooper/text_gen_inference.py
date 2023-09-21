"""Zero shot classification with generative language models."""
import asyncio
import warnings
from typing import Iterable

import aiohttp
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from thefuzz import process

__all__ = ["TGIZeroShotClassifier"]

default_prompt = """
### System:
You are a classification model that is really good at following
instructions and produces brief answers
that users can use as data right away.
Please follow the user's instructions as precisely as you can.
### User:
Your task will be to classify a text document into one
of the following classes: {classes}.
Please respond with a single label that you think fits
the document best.
Classify the following piece of text:
'{X}'
### Assistant:
"""


def async_run_all(requests: list[dict]):
    """Runs all requests asyncronously."""

    async def get_all(requests: list[dict]):
        async with aiohttp.ClientSession() as session:

            async def fetch(request: dict):
                async with session.post(**request) as response:
                    return await response.json()

            return await asyncio.gather(*[fetch(req) for req in requests])

    return asyncio.run(get_all(requests))


class TGIZeroShotClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible zero shot classification
    with HuggingFace's Text Generation Inference API.

    Parameters
    ----------
    endpoint: str, default 'http://127.0.0.1:8080/generate'
        Enpoint where the API is exposed. The default is localhost:8080.
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

    Attributes
    ----------
    classes_: array of str
        Class names learned from the labels.
    """

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8080/generate",
        prompt: str = default_prompt,
        max_new_tokens: int = 256,
        fuzzy_match: bool = True,
        device: str = "cpu",
    ):
        self.endpoint = endpoint
        self.prompt = prompt
        self.device = device
        self.classes_ = None
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

    def _generate_request(self, data: dict[str, str]) -> dict:
        prompt = self.prompt.format(**data)
        request_body = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": self.max_new_tokens},
        }
        return dict(
            url=self.endpoint,
            headers={"Content-Type": "application/json"},
            json=request_body,
        )

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
            raise NotFittedError("No class labels have been learnt yet.")
        requests = []
        classes_str = ", ".join([f"'{label}'" for label in self.classes_])
        for text in X:
            prompt_data = dict(X=text, classes=classes_str)
            requests.append(self._generate_request(prompt_data))
        responses = async_run_all(requests)
        labels = []
        for i, response in enumerate(responses):
            if "error" in response:
                error = response["error"]
                warnings.warn(
                    f"An error occurred for request {i}\n"
                    f"```{error}```\n"
                    "Assigning None."
                )
                labels.append(None)
                continue
            label = response["generated_text"].strip()
            if self.fuzzy_match and label not in self.classes_:
                label, _ = process.extractOne(label, self.classes_)  # type: ignore
            labels.append(label)
        return np.array(labels)


fewshot_prompt = """
### System:
You are a classification model that is really good at following
instructions and produces brief answers
that users can use as data right away.
Please follow the user's instructions as precisely as you can.
### User:
Your task will be to classify a text document into one
of the following classes: {classes}.
Please respond with a single label that you think fits
the document best.
Here are a couple of examples of labels assigned by experts:
{examples}
Classify the following piece of text:
'{X}'
### Assistant:
"""

example_prompt = """
Examples of texts labelled '{label}':
{examples}
"""


class TGIFewShotClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible few shot classification
    with HuggingFace's Text Generation Inference API.

    Parameters
    ----------
    endpoint: str, default 'http://127.0.0.1:8080/generate'
        Enpoint where the API is exposed. The default is localhost:8080.
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

    Attributes
    ----------
    classes_: array of str
        Class names learned from the labels.
    """

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8080/generate",
        prompt: str = fewshot_prompt,
        max_new_tokens: int = 256,
        fuzzy_match: bool = True,
        device: str = "cpu",
    ):
        self.endpoint = endpoint
        self.prompt = prompt
        self.device = device
        self.classes_ = None
        self.max_new_tokens = max_new_tokens
        self.fuzzy_match = fuzzy_match

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

    def _generate_request(self, data: dict[str, str]) -> dict:
        prompt = self.prompt.format(**data)
        request_body = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": self.max_new_tokens},
        }
        return dict(
            url=self.endpoint,
            headers={"Content-Type": "application/json"},
            json=request_body,
        )

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
            raise NotFittedError("No class labels have been learnt yet.")
        requests = []
        classes_str = ", ".join([f"'{label}'" for label in self.classes_])
        text_examples = []
        for label, examples in self.examples_.items():
            examples = [f"'{example}'" for example in examples]
            subprompt = example_prompt.format(
                label=label, examples="\n".join(examples)
            )
            text_examples.append(subprompt)
        examples_subprompt = "\n".join(text_examples)
        for text in X:
            prompt_data = dict(
                X=text, classes=classes_str, examples=examples_subprompt
            )
            requests.append(self._generate_request(prompt_data))
        responses = async_run_all(requests)
        labels = []
        for i, response in enumerate(responses):
            if "error" in response:
                error = response["error"]
                warnings.warn(
                    f"An error occurred for request {i}\n"
                    f"```{error}```\n"
                    "Assigning None."
                )
                labels.append(None)
                continue
            label = response["generated_text"].strip()
            if self.fuzzy_match and label not in self.classes_:
                label, _ = process.extractOne(label, self.classes_)  # type: ignore
            labels.append(label)
        return np.array(labels)

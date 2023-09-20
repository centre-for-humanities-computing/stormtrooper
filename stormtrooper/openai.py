import asyncio
import os
import time
from datetime import datetime
from typing import Iterable

import numpy as np
import openai
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from thefuzz import process
from tqdm import tqdm

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
 ```{X}```
### Assistant:
"""

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


def parse_prompt(prompt_template: str) -> dict[str, str]:
    """Parses prompt from string template."""
    prompt_parts = prompt_template.split("###")
    prompt_parts = [part.strip() for part in prompt_parts]
    prompt_parts = [part for part in prompt_parts if part]
    prompt_mapping = {}
    for part in prompt_parts:
        role, content = part.split(":", 1)
        prompt_mapping[role] = content
    if set(prompt_mapping.keys()) != {"System", "User", "Assistant"}:
        raise ValueError(
            "Prompts should have a 'System', 'User' and 'Assistant'"
            "section in OpenAI models."
        )
    return prompt_mapping


def create_messages(prompt: dict[str, str], data: dict[str, str]):
    """Produces messages to send to the chat API
    from prompt and data to infuse."""
    messages = [
        {"role": "system", "content": prompt["System"]},
        {"role": "user", "content": prompt["User"].format(**data)},
    ]
    return messages


class RequestHandler:
    """Utility class for handling requests asyncronously."""

    def __init__(self, max_retries: int = 5):
        self.current_minute = None
        self.max_retries = max_retries
        self.response_queue = []

    def wait_for_next_minute(self):
        """Waits for next minute before starting a new request."""
        if self.current_minute is None:
            self.current_minute = datetime.now().minute
        while datetime.now().minute == self.current_minute:
            time.sleep(1)
        self.current_minute = datetime.now().minute

    async def run_request(self, request: dict):
        """Starts a single request asyncronously."""
        res = None
        rate_error = None
        # Do N retries
        for retry in range(self.max_retries):
            try:
                self.current_minute = datetime.now().minute
                res = await openai.ChatCompletion.acreate(**request)
                self.response_queue.append(res)
                # We break out of the retry loop if the completion arrives
                # intact
                break
            except openai.error.RateLimitError as e:
                # If we exceed the rate limit we wait for the next minute.
                self.wait_for_next_minute()
                rate_error = e
        if res is None:
            # If we still didn't manage to get a response we propagate
            # the error to top level.
            raise openai.error.RateLimitError(
                "Maximal rate limit exceeded, "
                f"{self.max_retries} retries did not fix the issue."
            ) from rate_error

    async def run_requests(self, requests: Iterable[dict]) -> list:
        """Runs all requests simultaneously."""
        self.response_queue = []
        for request in requests:
            await self.run_request(request)
        return self.response_queue


class OpenAIZeroShotClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible zero shot classification
    with OpenAI's chat language models.

    Parameters
    ----------
    model_name: str, default 'gpt-3.5-turbo'
        Name of OpenAI chat model.
    temperature: float = 1.0
        What sampling temperature to use, between 0 and 2.
        Higher values like 0.8 will make the output more random,
        while lower values like 0.2 will make it
        more focused and deterministic.
    prompt: str, optional
        You can specify the prompt which will be used to prompt the model.
        Use placeholders to indicate where the class labels and the
        data should be placed in the prompt.
    max_new_tokens: int, default 256
        Maximum number of tokens the model should generate.
    max_retries: int, default 5
        Maximum number of retries if the rate limit is exceeded.
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
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 1.0,
        prompt: str = default_prompt,
        max_new_tokens: int = 256,
        max_retries: int = 5,
        fuzzy_match: bool = True,
        progress_bar: bool = True,
    ):
        self.model_name = model_name
        self.prompt = prompt
        self.temperature = temperature
        self.prompt_mapping_ = parse_prompt(prompt)
        self.classes_ = None
        self.max_new_tokens = max_new_tokens
        self.fuzzy_match = fuzzy_match
        self.progress_bar = progress_bar
        self.max_retries = max_retries
        try:
            openai.api_key = os.environ["OPENAI_API_KEY"]
            openai.organization = os.environ.get("OPENAI_ORG")
        except KeyError as e:
            raise KeyError(
                "Environment variable OPENAI_API_KEY not specified."
            ) from e

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

    def _produce_requests(self, X: Iterable[str]) -> Iterable[dict]:
        if self.classes_ is None:
            raise NotFittedError(
                "Class labels have not been collected yet, please fit."
            )
        if self.progress_bar:
            X = tqdm(X)
        classes_str = ", ".join([f"'{label}'" for label in self.classes_])
        for text in X:
            prompt_data = dict(X=text, classes=classes_str)
            messages = create_messages(self.prompt_mapping_, prompt_data)
            request = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_new_tokens,
            }
            yield request

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
        handler = RequestHandler(self.max_retries)
        requests = self._produce_requests(X)
        responses = asyncio.run(handler.run_requests(requests))
        results = []
        for response in responses:
            label = response["choices"][0]["message"]["content"].strip()
            if self.fuzzy_match and label not in self.classes_:
                label, _ = process.extractOne(label, self.classes_)
            results.append(label)
        return np.array(results)


class OpenAIFewShotClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible few shot classification
    with OpenAI's chat language models.

    Parameters
    ----------
    model_name: str, default 'gpt-3.5-turbo'
        Name of OpenAI chat model.
    temperature: float = 1.0
        What sampling temperature to use, between 0 and 2.
        Higher values like 0.8 will make the output more random,
        while lower values like 0.2 will make it
        more focused and deterministic.
    prompt: str, optional
        You can specify the prompt which will be used to prompt the model.
        Use placeholders to indicate where the class labels and the
        data should be placed in the prompt.
    max_new_tokens: int, default 256
        Maximum number of tokens the model should generate.
    max_retries: int, default 5
        Maximum number of retries if the rate limit is exceeded.
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
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 1.0,
        prompt: str = fewshot_prompt,
        max_new_tokens: int = 256,
        max_retries: int = 5,
        fuzzy_match: bool = True,
        progress_bar: bool = True,
    ):
        self.model_name = model_name
        self.prompt = prompt
        self.temperature = temperature
        self.prompt_mapping_ = parse_prompt(prompt)
        self.classes_ = None
        self.max_new_tokens = max_new_tokens
        self.fuzzy_match = fuzzy_match
        self.progress_bar = progress_bar
        self.max_retries = max_retries
        try:
            openai.api_key = os.environ["OPENAI_API_KEY"]
            openai.organization = os.environ.get("OPENAI_ORG")
        except KeyError as e:
            raise KeyError(
                "Environment variable OPENAI_API_KEY not specified."
            ) from e

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

    def _produce_requests(self, X: Iterable[str]) -> Iterable[dict]:
        if self.classes_ is None:
            raise NotFittedError(
                "Class labels have not been collected yet, please fit."
            )
        if self.progress_bar:
            X = tqdm(X)
        text_examples = []
        for label, examples in self.examples_.items():
            examples = [f"'{example}'" for example in examples]
            subprompt = example_prompt.format(
                label=label, examples="\n".join(examples)
            )
            text_examples.append(subprompt)
        examples_subprompt = "\n".join(text_examples)
        classes_str = ", ".join([f"'{label}'" for label in self.classes_])
        for text in X:
            prompt_data = dict(
                X=text, classes=classes_str, examples=examples_subprompt
            )
            messages = create_messages(
                self.prompt_mapping_,
                prompt_data,
            )
            request = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_new_tokens,
            }
            yield request

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
        handler = RequestHandler(self.max_retries)
        requests = self._produce_requests(X)
        responses = asyncio.run(handler.run_requests(requests))
        results = []
        for response in responses:
            label = response["choices"][0]["message"]["content"].strip()
            if self.fuzzy_match and label not in self.classes_:
                label, _ = process.extractOne(label, self.classes_)
            results.append(label)
        return np.array(results)

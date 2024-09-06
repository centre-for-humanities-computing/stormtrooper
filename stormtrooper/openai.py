import asyncio
import os
from typing import Iterable

import numpy as np
import openai
from sklearn.exceptions import NotFittedError
from tqdm import tqdm

from stormtrooper.chat import (ChatClassifier, default_prompt,
                               default_system_prompt)


class OpenAIClassifier(ChatClassifier):
    """Use OpenAI's models for zero and few-shot text classification.

    Parameters
    ----------
    model_name: str, default "gpt-3.5-turbo"
        Name of the OpenAI chat model to use.
    temperature: float, default 1.0
        Temperature for text generation.
        Higher temperature results in more diverse answers.
    prompt: str
        Prompt template to use for each text.
    system_prompt: str
        System prompt for the model.
    max_new_tokens: int, default 256
        Maximum number of new tokens to generate.
    fuzzy_match: bool, default True
        Indicates whether responses should be fuzzy-matched to closest learned label.
    progress_bar: bool, default True
        Inidicates whether a progress bar should be desplayed when obtaining results.
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 1.0,
        prompt: str = default_prompt,
        system_prompt: str = default_system_prompt,
        max_new_tokens: int = 256,
        fuzzy_match: bool = True,
        progress_bar: bool = True,
    ):
        self.model_name = model_name
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.classes_ = None
        self.max_new_tokens = max_new_tokens
        self.fuzzy_match = fuzzy_match
        try:
            openai.api_key = os.environ["OPENAI_API_KEY"]
            openai.organization = os.environ.get("OPENAI_ORG")
            client = openai.OpenAI(api_key=openai.api_key)
            valid_model_ids = [model.id for model in client.models.list()]
            if model_name not in valid_model_ids:
                raise ValueError(
                    f"{model_name} is not a valid model ID for OpenAI."
                )
        except KeyError as e:
            raise KeyError(
                "Environment variable OPENAI_API_KEY not specified."
            ) from e
        self.client = openai.AsyncOpenAI()

    async def predict_one_async(self, text: str) -> str:
        messages = self.generate_messages(text)
        response = await self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        return response.choices[0].message.content

    def predict_one(self, text: str) -> str:
        return asyncio.run(self.predict_one_async(text))

    async def predict_async(self, X: Iterable[str]) -> np.ndarray:
        if self.classes_ is None:
            raise NotFittedError(
                "Class labels have not been collected yet, please fit."
            )
        if self.progress_bar:
            X = tqdm(X)
        res = await asyncio.gather(*[self.predict_one_async(x) for x in X])
        return np.array(res)

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
        return asyncio.run(self.predict_async(X))

"""Zero shot classification with generative language models."""

from typing import Optional

import numpy as np
from transformers import pipeline

from stormtrooper.chat import (ChatClassifier, default_prompt,
                               default_system_prompt)

__all__ = ["GenerativeClassifier"]


class GenerativeClassifier(ChatClassifier):
    """Scikit-learn compatible zero shot classification
    with generative language models.

    Parameters
    ----------
    model_name: str, default 'HuggingFaceH4/zephyr-7b-beta'
        Generative instruct model from HuggingFace.
    prompt: str, optional
        You can specify the prompt which will be used to prompt the model.
        Use placeholders to indicate where the class labels and the
        data should be placed in the prompt.
    system_prompt: str, optional
        System prompt for the model.
    max_new_tokens: int, default 256
        Maximum number of tokens the model should generate.
    fuzzy_match: bool, default True
        Indicates whether the output lables should be fuzzy matched
        to the learnt class labels.
        This is useful when the model isn't giving specific enough answers.
    progress_bar: bool, default True
        Indicates whether a progress bar should be shown.
    device: str, default None
        Indicates which device should be used for classification.
        Models are by default run on CPU.
    device_map: str, default None
        Device map argument for very large models.

    Attributes
    ----------
    classes_: array of str
        Class names learned from the labels.
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceH4/zephyr-7b-beta",
        prompt: str = default_prompt,
        system_prompt: str = default_system_prompt,
        max_new_tokens: int = 256,
        fuzzy_match: bool = True,
        progress_bar: bool = True,
        device: Optional[str] = None,
        device_map: Optional[str] = None,
    ):
        self.model_name = model_name
        self.prompt = prompt
        self.device = device
        self.system_prompt = system_prompt
        self.device_map = device_map
        self.pipeline = pipeline(
            "text-generation",
            self.model_name,
            device=self.device,
            device_map=self.device_map,
        )
        self.classes_ = None
        self.max_new_tokens = max_new_tokens
        self.fuzzy_match = fuzzy_match
        self.progress_bar = progress_bar

    def predict_one(self, text: str) -> np.ndarray:
        messages = self.generate_messages(text)
        response = self.pipeline(messages, max_new_tokens=self.max_new_tokens)[
            0
        ]["generated_text"][-1]
        label = response["content"]
        return label

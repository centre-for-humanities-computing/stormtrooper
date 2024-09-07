"""Zero shot classification with text-to-text language models."""

from typing import Optional

from transformers import pipeline

from stormtrooper.chat import ChatClassifier, default_prompt

__all__ = ["Text2TextClassifier"]


class Text2TextClassifier(ChatClassifier):
    """Zero and few-shot classification
    with text2text language models.

    Parameters
    ----------
    model_name: str, default 'google/flan-t5-base'
        Text2text model from HuggingFace.
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
        model_name: str = "google/flan-t5-base",
        prompt: str = default_prompt,
        max_new_tokens: int = 256,
        fuzzy_match: bool = True,
        progress_bar: bool = True,
        device: Optional[str] = None,
        device_map: Optional[str] = None,
    ):
        self.model_name = model_name
        self.prompt = prompt
        self.device = device
        self.device_map = device_map
        self.pipeline = pipeline(
            "text2text-generation",
            model=model_name,
            device=device,
            device_map=device_map,
        )
        self.classes_ = None
        self.progress_bar = progress_bar
        self.max_new_tokens = max_new_tokens
        self.fuzzy_match = fuzzy_match

    def predict_one(self, text: str) -> str:
        prompt = self.get_user_prompt(text)
        response = self.pipeline(prompt)
        return response[0]["generated_text"]

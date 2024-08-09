"""Zero shot classification with text-to-text language models."""

from transformers import pipeline

from stormtrooper.chat import ChatClassifier, default_prompt

__all__ = ["Text2TextClassifier"]


class Text2TextClassifier(ChatClassifier):
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
        self.pipeline = pipeline(
            "text2text-generation", model=model_name, device=device
        )
        self.classes_ = None
        self.progress_bar = progress_bar
        self.max_new_tokens = max_new_tokens
        self.fuzzy_match = fuzzy_match

    def predict_one(self, text: str) -> str:
        prompt = self.get_user_prompt(text)
        response = self.pipeline(prompt)
        return response[0]["generated_text"]

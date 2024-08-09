import os

import openai

from stormtrooper.chat import (ChatClassifier, default_prompt,
                               default_system_prompt)


class OpenAIClassifier(ChatClassifier):
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
                raise ValueError(f"{model_name} is not a valid model ID for OpenAI.")
        except KeyError as e:
            raise KeyError("Environment variable OPENAI_API_KEY not specified.") from e
        self.client = openai.OpenAI()

    def predict_one(self, text: str) -> str:
        messages = self.generate_messages(text)
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        return response.choices[0].message.content

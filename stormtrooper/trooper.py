from typing import Iterable, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from transformers import AutoConfig
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES)

from stormtrooper.error import NotInstalled
from stormtrooper.generative import GenerativeClassifier
from stormtrooper.set_fit import SetFitClassifier
from stormtrooper.text2text import Text2TextClassifier
from stormtrooper.zero_shot import ZeroShotClassifier

try:
    from stormtrooper.openai import OpenAIClassifier
except ModuleNotFoundError:
    OpenAIClassifier = NotInstalled("OpenAIClassifier", "openai")


def is_text2text(config) -> bool:
    for architecture in config.architectures:
        if (
            architecture
            in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES.values()
        ):
            return True
    return False


def is_generative(config) -> bool:
    for architecture in config.architectures:
        if architecture in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            return True
    return False


def is_nli(config) -> bool:
    for architecture in config.architectures:
        if (
            architecture
            in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.values()
        ):
            return True
    return False


model_type_to_cls = {
    "text2text": Text2TextClassifier,
    "generative": GenerativeClassifier,
    "nli": ZeroShotClassifier,
    "setfit": SetFitClassifier,
    "openai": OpenAIClassifier,
}


def get_model_type(model_name: str) -> str:
    try:
        config = AutoConfig.from_pretrained(model_name)
        if is_text2text(config):
            return "text2text"
        elif is_generative(config):
            return "generative"
        elif is_nli(config):
            return "nli"
        else:
            return "setfit"
    except OSError as e:
        try:
            config = AutoConfig.from_pretrained(
                "sentence-transformers/" + model_name
            )
            return "setfit"
        except OSError as e:
            try:
                model = OpenAIClassifier(model_name)
                return "openai"
            except (KeyError, ValueError) as e2:
                raise ValueError(
                    f"Model {model_name} cannot be found in HuggingFace repositories, nor could an OpenAI model be initialized."
                ) from e2


class Trooper(BaseEstimator, ClassifierMixin):
    """Generic zero-shot, few-shot classifier.
    Automatically determines the type based on model name.

    Parameters
    ----------
    model_name: str
        Name of the base model to use.
        Could be a model from OpenAI or HuggingFace Hub.
    progress_bar: bool, default True
        Indicates whether a progress bar should be displayed during inference.
    device: str, default None
        The device the model should run on (only valid for locally run models).
    device_map: str, default None
        Device map argument for very large models.
    prompt: str, default None
        Prompt to use for promptable models.
    system_prompt: str, default None
        System prompt to use for chat models.
    fuzzy_match: bool, default True
        Indicates whether responses should be fuzzy matched to the closest class name,
        when using a generative model.
        This is useful, as large language models sometimes misspell words or capitalize them
        when responding to queries.
        With fuzzy matching you can rest assured that you will only get responses exactly
        matching target labels.
    """

    def __init__(
        self,
        model_name: str,
        *,
        progress_bar: bool = True,
        device: Optional[str] = None,
        device_map: Optional[str] = None,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        fuzzy_match: bool = True,
    ):
        self.model_name = model_name
        self.model_type = get_model_type(model_name)
        self.progress_bar = progress_bar
        self.device_map = device_map
        self.device = device
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.fuzzy_match = fuzzy_match
        model_kwargs = dict(model_name=model_name)
        if self.model_type in ["generative", "openai", "text2text"]:
            model_kwargs["fuzzy_match"] = self.fuzzy_match
            if self.prompt is not None:
                model_kwargs["prompt"] = self.prompt
            if (
                self.model_type == "text2text"
            ) and self.system_prompt is not None:
                model_kwargs["system_prompt"] = self.system_prompt
        if self.model_type in ["generative", "text2text", "nli", "setfit"]:
            model_kwargs["device"] = self.device
        if self.model_type in ["generative", "text2text"]:
            model_kwargs["device_map"] = self.device_map
        if self.model_type in ["nli", "generative", "text2text", "openai"]:
            model_kwargs["progress_bar"] = self.progress_bar
        self.model = model_type_to_cls[self.model_type](**model_kwargs)

    def fit(self, X: Optional[Iterable[str]], y: Iterable[str]):
        """Learns class labels and potential examples.

        Parameters
        ----------
        X: iterable of str
            Examples to use in few-shot prompts.
            Pass None, when no examples are to be used.
        y: iterable of str
            Class labels.
        """
        self.model.fit(X, y)
        return self

    def partial_fit(self, X: Optional[Iterable[str]], y: Iterable[str]):
        """Learns class labels and potential examples in a batch.

        Parameters
        ----------
        X: iterable of str
            Examples to use in few-shot prompts.
            Pass None, when no examples are to be used.
        y: iterable of str
            Class labels.
        """
        self.model.partial_fit(X, y)
        return self

    def predict(self, X: Iterable[str]) -> np.ndarray:
        """Predicts labels for a set of examples

        Parameters
        ----------
        X: iterable of str
            Documents to predict labels for.

        Returns
        -------
        ndarray of shape (n_documents,)
            Labels for documents.
        """
        return self.model.predict(X)

    def predict_proba(self, X: Iterable[str]) -> np.ndarray:
        """Predicts probability of each label.
        Only available for certain models.

        Parameters
        ----------
        X: iterable of str
            Documents to predict labels for.

        Returns
        -------
        ndarray of shape (n_documents, n_labels)
            Label distributions for documents.
        """
        return self.model.predict_proba(X)

    @property
    def classes_(self):
        return self.model.classes

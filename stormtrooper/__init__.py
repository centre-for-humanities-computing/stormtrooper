from stormtrooper.error import NotInstalled
from stormtrooper.generative import (GenerativeFewShotClassifier,
                                     GenerativeZeroShotClassifier)
from stormtrooper.text2text import (Text2TextFewShotClassifier,
                                    Text2TextZeroShotClassifier)
from stormtrooper.text_gen_inference import (TGIFewShotClassifier,
                                             TGIZeroShotClassifier)
from stormtrooper.zero_shot import ZeroShotClassifier

try:
    from stormtrooper.set_fit import (SetFitFewShotClassifier,
                                      SetFitZeroShotClassifier)
except ModuleNotFoundError:
    SetFitZeroShotClassifier = NotInstalled(
        "SetFitZeroShotClassifier", "setfit"
    )
    SetFitFewShotClassifier = NotInstalled("SetFitFewShotClassifier", "setfit")

try:
    from stormtrooper.openai import (OpenAIFewShotClassifier,
                                     OpenAIZeroShotClassifier)
except ModuleNotFoundError:
    OpenAIZeroShotClassifier = NotInstalled(
        "OpenAIZeroShotClassifier", "openai"
    )
    OpenAIFewShotClassifier = NotInstalled("OpenAIFewShotClassifier", "openai")

__all__ = [
    "GenerativeZeroShotClassifier",
    "GenerativeFewShotClassifier",
    "Text2TextZeroShotClassifier",
    "Text2TextFewShotClassifier",
    "ZeroShotClassifier",
    "SetFitZeroShotClassifier",
    "SetFitFewShotClassifier",
    "OpenAIFewShotClassifier",
    "OpenAIZeroShotClassifier",
    "TGIZeroShotClassifier",
    "TGIFewShotClassifier",
]

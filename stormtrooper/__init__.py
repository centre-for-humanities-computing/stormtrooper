from stormtrooper.error import NotInstalled
from stormtrooper.generative import GenerativeClassifier
from stormtrooper.text2text import Text2TextClassifier
from stormtrooper.text_gen_inference import TGIFewShotClassifier, TGIZeroShotClassifier
from stormtrooper.trooper import Trooper
from stormtrooper.zero_shot import ZeroShotClassifier

try:
    from stormtrooper.set_fit import SetFitClassifier
except ModuleNotFoundError:
    SetFitClassifier = NotInstalled("SetFitClassifier", "setfit")

try:
    from stormtrooper.openai import OpenAIClassifier
except ModuleNotFoundError:
    OpenAIClassifier = NotInstalled("OpenAIClassifier", "openai")

__all__ = [
    "GenerativeClassifier",
    "OpenAIClassifier",
    "SetFitClassifier",
    "Text2TextClassifier",
    "ZeroShotClassifier",
    "TGIZeroShotClassifier",
    "TGIFewShotClassifier",
    "Trooper",
]

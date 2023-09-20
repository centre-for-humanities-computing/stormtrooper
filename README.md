<img align="left" width="82" height="82" src="assets/logo.svg">

# stormtrooper

<br>
Transformer-based zero/few shot learning components for scikit-learn pipelines.

[Documentation](https://centre-for-humanities-computing.github.io/stormtrooper/)

## New in version 0.4.0 :fire:

- You can now use OpenAI's chat models with blazing fast :zap: async inference.

## New in version 0.3.0 🌟 

- SetFit is now part of the library and can be used in scikit-learn workflows.

## Example

```bash
pip install stormtrooper
```

```python
class_labels = ["atheism/christianity", "astronomy/space"]
example_texts = [
    "God came down to earth to save us.",
    "A new nebula was recently discovered in the proximity of the Oort cloud."
]
```


### Zero-shot learning

For zero-shot learning you can use zero-shot models:
```python
from stormtrooper import ZeroShotClassifier
classifier = ZeroShotClassifier().fit(None, class_labels)
```

Generative models (GPT, Llama):
```python
from stormtrooper import GenerativeZeroShotClassifier
# You can hand-craft prompts if it suits you better, but
# a default prompt is already available
prompt = """
### System:
You are a literary expert tasked with labeling texts according to
their content.
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
classifier = GenerativeZeroShotClassifier(prompt=prompt).fit(None, class_labels)
```

Text2Text models (T5):
If you are running low on resources I would personally recommend T5.
```python
from stormtrooper import Text2TextZeroShotClassifier
# You can define a custom prompt, but a default one is available
prompt = "..."
classifier =Text2TextZeroShotClassifier(prompt=prompt).fit(None, class_labels)
```

```python
predictions = classifier.predict(example_texts)

assert list(predictions) == ["atheism/christianity", "astronomy/space"]
```

OpenAI models:
You can now use OpenAI's chat LLMs in stormtrooper workflows.

```python
from stormtrooper import OpenAIZeroShotClassifier

classifier = OpenAIZeroShotClassifier("gpt-4").fit(None, class_labels)
```

```python
predictions = classifier.predict(example_texts)

assert list(predictions) == ["atheism/christianity", "astronomy/space"]
```

### Few-Shot Learning

For few-shot tasks you can only use Generative, Text2Text, OpenAI (aka. promptable) or SetFit models.

```python
from stormtrooper import GenerativeFewShotClassifier, Text2TextFewShotClassifier, SetFitFewShotClassifier

classifier = SetFitFewShotClassifier().fit(example_texts, class_labels)
predictions = model.predict(["Calvinists believe in predestination."])

assert list(predictions) == ["atheism/christianity"]
```

### Fuzzy Matching

Generative and text2text models by default will fuzzy match results to the closest class label, you can disable this behavior
by specifying `fuzzy_match=False`.

If you want fuzzy matching speedup, you should install `python-Levenshtein`.

### Inference on GPU

From version 0.2.2 you can run models on GPU.
You can specify the device when initializing a model:

```python
classifier = Text2TextZeroShotClassifier(device="cuda:0")
```

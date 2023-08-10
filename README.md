<img align="left" width="82" height="82" src="assets/logo.svg">

# stormtrooper

<br>
Transformer-based zero/few shot learning components for scikit-learn pipelines.

## Example

```bash
pip install stormtrooper
```

### Zero shot models

```python
from stormtrooper import ZeroShotClassifier

class_labels = ["atheism/christianity", "astronomy/space"]
classifier = ZeroShotClassifier().fit(None, class_labels)

example_texts = [
    "God came down to earth to save us.",
    "A new nebula was recently discovered in the proximity of the Oort cloud."
]
predictions = classifier.predict(example_texts)

assert list(predictions) == ["atheism/christianity", "astronomy/space"]
```

### Text2Text models (T5)

```python
from stormtrooper import Text2TextZeroShotClassifier

class_labels = ["atheism/christianity", "astronomy/space"]
classifier = Text2TextZeroShotClassifier().fit(None, class_labels)

example_texts = [
    "God came down to earth to save us.",
    "A new nebula was recently discovered in the proximity of the Oort cloud."
]
predictions = classifier.predict(example_texts)

assert list(predictions) == ["atheism/christianity", "astronomy/space"]
```

### Text generation models (Llama/GPT)

```python
from stormtrooper import GenerativeZeroShotClassifier

class_labels = ["atheism/christianity", "astronomy/space"]
classifier = GenerativeZeroShotClassifier().fit(None, class_labels)

example_texts = [
    "God came down to earth to save us.",
    "A new nebula was recently discovered in the proximity of the Oort cloud."
]
predictions = classifier.predict(example_texts)

assert list(predictions) == ["atheism/christianity", "astronomy/space"]
```

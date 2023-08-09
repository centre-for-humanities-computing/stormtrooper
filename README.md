# stormtrooper
Transformer-based zero/few shot learning components for scikit-learn pipelines.

## Example

```bash
pip install stormtrooper
```

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

<img align="left" width="82" height="82" src="assets/logo.svg">

# stormtrooper

<br>
Zero/few shot learning components for scikit-learn pipelines with large-language models and transformers.

[Documentation](https://centre-for-humanities-computing.github.io/stormtrooper/)

## New in 1.0.0

### `Trooper`
The brand new `Trooper` interface allows you not to have to specify what model type you wish to use.
Stormtrooper will automatically detect the model type from the specified name.

```python
from stormtrooper import Trooper

# This loads a setfit model
model = Trooper("all-MiniLM-L6-v2")

# This loads an OpenAI model
model = Trooper("gpt-4")

# This loads a Text2Text model
model = Trooper("google/flan-t5-base")
```

### Unified zero and few-shot classification

You no longer have to specify whether a model should be a few or a zero-shot classifier when initialising it.
If you do not pass any training examples, it will be automatically assumed that the model should be zero-shot.

```python
# This is a zero-shot model
model.fit(None, ["dog", "cat"])

# This is a few-shot model
model.fit(["he was a good boy", "just lay down on my laptop"], ["dog", "cat"])

```
## Model types

You can use all sorts of transformer models for few and zero-shot classification in Stormtrooper.

1. Instruction fine-tuned generative models, e.g. `Trooper("HuggingFaceH4/zephyr-7b-beta")`
2. Encoder models with SetFit, e.g. `Trooper("all-MiniLM-L6-v2")`
3. Text2Text models e.g. `Trooper("google/flan-t5-base")`
4. OpenAI models e.g. `Trooper("gpt-4")`
5. NLI models e.g. `Trooper("facebook/bart-large-mnli")`

## Example usage

Find more in our [docs](https://centre-for-humanities-computing.github.io/stormtrooper/).

```bash
pip install stormtrooper
```

```python
from stormtrooper import Trooper

class_labels = ["atheism/christianity", "astronomy/space"]
example_texts = [
    "God came down to earth to save us.",
    "A new nebula was recently discovered in the proximity of the Oort cloud."
]
new_texts = ["God bless the reailway workers", "The frigate is ready to launch from the spaceport"]

# Zero-shot classification
model = Trooper("google/flan-t5-base")
model.fit(None, class_labels)
model.predict(new_texts)
# ["atheism/christianity", "astronomy/space"]

# Few-shot classification
model = Trooper("google/flan-t5-base")
model.fit(example_texts, class_labels)
model.predict(new_texts)
# ["atheism/christianity", "astronomy/space"]
```

### Fuzzy Matching

Generative and text2text models by default will fuzzy match results to the closest class label, you can disable this behavior
by specifying `fuzzy_match=False`.

If you want fuzzy matching speedup, you should install `python-Levenshtein`.

### Inference on GPU

From version 0.2.2 you can run models on GPU.
You can specify the device when initializing a model:

```python
classifier = Trooper("all-MiniLM-L6-v2", device="cuda:0")
```

### Inference on multiple GPUs

You can run a model on multiple devices in order of device priority `GPU -> CPU + Ram -> Disk` and on multiple devices by using the `device_map` argument.
Note that this only works with text2text and generative models.

```
model = Trooper("HuggingFaceH4/zephyr-7b-beta", device_map="auto")
```

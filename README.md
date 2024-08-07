<img align="left" width="82" height="82" src="assets/logo.svg">

# stormtrooper

<br>
Zero/few shot learning components for scikit-learn pipelines with large-language models and transformers.

[Documentation](https://centre-for-humanities-computing.github.io/stormtrooper/)

## Why stormtrooper?

Other packages promise to provide at least similar functionality (scikit-llm), why should you choose stormtrooper instead?

1. Fine-grained control over you pipeline.
    - Variety: stormtrooper allows you to use virtually all canonical approaches for zero and few-shot classification including NLI, Seq2Seq and Generative open-access models from Transformers, SetFit and even OpenAI's large language models.
    - Prompt engineering: You can adjust prompt templates to your hearts content.
2. Performance
    - Easy inference on GPU if you have access to it.
    - Interfacing HuggingFace's TextGenerationInference API, the most efficient way to host models locally.
    - Async interaction with external APIs, this can speed up inference with OpenAI's models quite drastically. 
3. Extensive [Documentation](https://centre-for-humanities-computing.github.io/stormtrooper/)
   - Throrough API reference and loads of examples to get you started.
3. Battle-hardened
    - We at the Center For Humanities Computing are making extensive use of this package. This means you can rest assured that the package works under real-world pressure. As such you can expect regular updates and maintance.
4. Simple
    - We opted for as bare-bones of an implementation and little coupling as possible. The library works at the lowest level of abstraction possible, and we hope our code will be rather easy for others to understand and contribute to.


## New in version 0.5.0

stormtrooper now uses chat templates from HuggingFace transformers for generative models.
This means that you no longer have to pass model-specific prompt templates to these and can define system and user prompts separately.

```python
from stormtrooper import GenerativeZeroShotClassifier

system_prompt = "You're a helpful assistant."
user_prompt = """
Classify a text into one of the following categories: {classes}
Text to clasify:
"{X}"
"""

model = GenerativeZeroShotClassifier().fit(None, ["political", "not political"])
model.predict("Joe Biden is no longer the candidate of the Democrats.")
```


## Examples

Here are a couple of motivating examples to get you hooked. Find more in our [docs](https://centre-for-humanities-computing.github.io/stormtrooper/).

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
classifier = GenerativeZeroShotClassifier("meta-llama/Meta-Llama-3.1-8B-Instruct").fit(None, class_labels)
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

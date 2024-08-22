
# Models

Stormtrooper supports a wide variety of model types for zero and few-shot classification.
This page includes a general overview of how the different methods approach this task.

## Text Generation

LLMs are a relatively easy to utilise for zero and few-shot classification, as they contain a lot of general language-based knowledge and can provide free-form answers.
Models that generate text typically have to be _prompted_. One has to pass free-form text instructions to a model, to which it can respond with a (hopefully) appropriate answer.

### Instruction Models

Models used in chatbots and alike are typically instruction-finetuned generatively pretrained transformer models.
These models take a string of messages and generate a new message at the end by predicting next-token probabilities.

These models also typically take a _system prompt_ a base prompt that tells the model what persona it should have and how it should behave when presented with instructions.

You can use instruction models from both HuggingFace Hub, but also OpenAI in stromtrooper.

```python
from stormtrooper import Trooper

# Model from HuggingFace:
model = Trooper("HuggingFaceH4/zephyr-7b-beta")

# OpenAI model
model = Trooper("gpt-4")
```

::: stormtrooper.openai.OpenAIClassifier

::: stormtrooper.generative.GenerativeClassifier

### Text2Text Models

Text2Text models not only generate text, but are trained to predict a sequence of text based on a sequence of incoming text.
Input text gets encoded into a low-dimensional latent space, and then this latent representation is used to generate an appropriate response, similar to an AutoEncoder.

Text2Text models are typically smaller and faster than fully generative models, but also less performant.

```python
from stormtrooper import Trooper

model = Trooper("google/flan-t5-small")
```
::: stormtrooper.text2text.Text2TextClassifier

## Sentence Transformers + SetFit

<figure>
  <img src="https://raw.githubusercontent.com/huggingface/setfit/main/assets/setfit.png" width="90%" style="margin-left: auto;margin-right: auto;">
  <figcaption>Schematic overview of SetFit</figcaption>
</figure>

SetFit is a commonly employed trick for training classifiers on a low number of labelled datapoints.
It involves:

1. Finetuning a sentence encoder model using contrastive loss, where positive pairs are the examples that belong in the same category, and negative pairs are documents belonging to different classes.
2. Training a classification head on the finetuned embeddings.

When you load any encoder-style model in Stormtrooper, they are automatically converted into a SetFit model.

```python
from stormtrooper import Trooper

model = Trooper("all-MiniLM-L6-v2")
```

::: stormtrooper.set_fit.SetFitClassifier

## Natural Language Inference

Natural language inference entails classifying pairs of texts based on whether they are congruent with each other.
Models finetuned for NLI can also be utilised for zero-shot classification.

```python
from stormtrooper import Trooper

model = Trooper("facebook/bart-large-mnli").fit(None, ["dog", "cat"])
model.predict_proba(["He was barking like hell"])
# array([[0.95, 0.005]])
```

::: stormtrooper.zero_shot.ZeroShotClassifier

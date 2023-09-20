OpenAI models
=================

Stormtrooper gives you access to OpenAI's chat models for zero and few-shot classification.
You get full control over temperature settings, system and user prompts.
In contrast to other packages, like scikit-llm, stormtrooper also uses Python's asyncio to concurrently
interact with OpenAI's API. This can give multiple times speedup on several tasks. 

You can also set upper limits for number of requests and tokens per minute, so you don't exceed your quota.
This is by default set to the limit of the payed tier on OpenAI's API.

You need to install stormtrooper with optional dependencies.

.. code-block:: bash

   pip install stormtrooper[openai]

You additionally need to set the OpenAI API key as an environment variable.

.. code-block:: bash

   export OPENAI_API_KEY="sk-..."
   # Setting organization is optional
   export OPENAI_ORG="org-..."

.. code-block:: python

   from stormtrooper import OpenAIZeroShotClassifier, OpenAIFewShotClassifier

   sample_text = "It is the Electoral College's responsibility to elect the president."

   labels = ["politics", "science", "other"]

Here's a zero shot example with ChatGPT 3.5:

.. code-block:: python

   model = OpenAIZeroShotClassifier("gpt-3.5-turbo").fit(None, labels)
   predictions = model.predict([sample_text])
   assert list(predictions) == ["politics"]
   
And a few shot example with ChatGPT 4:

.. code-block:: python

   few_shot_examples = [
     "Joe Biden is the president.",
     "Liquid water was found on the moon.",
     "Jerry likes football."
   ]

   model = OpenAIFewShotClassifier("gpt-4", temperature=0.2).fit(few_shot_examples, labels)
   predictions = model.predict([sample_text])
   assert list(predictions) == ["politics"]


The format of the prompts is the same as with StableBeluga instruct models, and an error is raised if your prompt does not follow
this format.

.. code-block:: python
  
   prompt = """
   ### System:
   You are a helpful assistant
   ### User:
   Your task will be to classify a text document into one
   of the following classes: {classes}.
   Please respond with a single label that you think fits
   the document best.
   Classify the following piece of text:
   '{X}'
   ### Assistant:
   """

   model = OpenAIZeroShotClassifier("gpt-4", prompt=prompt)


API reference
^^^^^^^^^^^^^

.. autoclass:: stormtrooper.OpenAIZeroShotClassifier
   :members:

.. autoclass:: stormtrooper.OpenAIFewShotClassifier
   :members:

Generative models
=================

stormtrooper also supports fully generative model architectures both for few-shot and zero-shot learning.
It's worth noting that most instruct finetuned generative models are quite hefty and take a lot of resources
to run. We recommend that you exhaust all other options before you turn to generative models.

.. code-block:: python

   from stormtrooper import GenerativeZeroShotClassifier, GenerativeFewShotClassifier

   sample_text = "It is the Electoral College's responsibility to elect the president."

   labels = ["politics", "science", "other"]

Here's a zero shot example:

.. code-block:: python

   model = GenerativeZeroShotClassifier("stabilityai/StableBeluga-13B").fit(None, labels)
   predictions = model.predict([sample_text])
   assert list(predictions) == ["politics"]
   
And a few shot example:

.. code-block:: python

   few_shot_examples = [
     "Joe Biden is the president.",
     "Liquid water was found on the moon.",
     "Jerry likes football."
   ]

   model = GenerativeFewShotClassifier("stabilityai/StableBeluga-13B").fit(few_shot_examples, labels)
   predictions = model.predict([sample_text])
   assert list(predictions) == ["politics"]


API reference
^^^^^^^^^^^^^

.. autoclass:: stormtrooper.GenerativeZeroShotClassifier
   :members:

.. autoclass:: stormtrooper.GenerativeFewShotClassifier
   :members:

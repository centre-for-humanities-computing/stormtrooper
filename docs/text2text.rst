Text2Text models
================

You can easily employ text2text models for zero and few-shot classification.
By default FLAN-T5 base is used.

.. code-block:: python

   from stormtrooper import Text2TextZeroShotClassifier, Text2TextFewShotClassifier

   sample_text = "It is the Electoral College's responsibility to elect the president."

   labels = ["politics", "science", "other"]

Here's a zero shot example:

.. code-block:: python

   model = Text2TextZeroShotClassifier().fit(None, labels)
   predictions = model.predict([sample_text])
   assert list(predictions) == ["politics"]
   
And a few shot example:

.. code-block:: python

   few_shot_examples = [
     "Joe Biden is the president.",
     "Liquid water was found on the moon.",
     "Jerry likes football."
   ]

   model = Text2TextFewShotClassifier().fit(few_shot_examples, labels)
   predictions = model.predict([sample_text])
   assert list(predictions) == ["politics"]


API reference
^^^^^^^^^^^^^

.. autoclass:: stormtrooper.Text2TextZeroShotClassifier
   :members:

.. autoclass:: stormtrooper.Text2TextFewShotClassifier
   :members:

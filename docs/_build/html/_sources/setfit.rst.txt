SetFit models
================

You can use `SetFit <https://github.com/huggingface/setfit>`_ in stormtrooper for training efficient zero and few-shot learning models from sentence transformers.

SetFit uses a prompt-free approach, needs way smaller models, and is thus faster to train, and more employable in high-performance settings for inference.

.. image:: https://raw.githubusercontent.com/huggingface/setfit/main/assets/setfit.png

Since this requires the setfit package we recommend you install stormtrooper with its optional dependencies.

.. code-block:: bash

   pip install stormtrooper[setfit]

.. code-block:: python

   from stormtrooper import SetFitZeroShotClassifier, SetFitFewShotClassifier

   sample_text = "It is the Electoral College's responsibility to elect the president."

   labels = ["politics", "science", "other"]

Here's a zero shot example:

.. code-block:: python

   model = SetFitZeroShotClassifier().fit(None, labels)
   predictions = model.predict([sample_text])
   assert list(predictions) == ["politics"]
   
And a few shot example:

.. code-block:: python

   few_shot_examples = [
     "Joe Biden is the president.",
     "Liquid water was found on the moon.",
     "Jerry likes football."
   ]

   model = SetFitFewShotClassifier().fit(few_shot_examples, labels)
   predictions = model.predict([sample_text])
   assert list(predictions) == ["politics"]


API reference
^^^^^^^^^^^^^

.. autoclass:: stormtrooper.setfit.SetFitZeroShotClassifier
   :members:

.. autoclass:: stormtrooper.setfit.SetFitFewShotClassifier
   :members:

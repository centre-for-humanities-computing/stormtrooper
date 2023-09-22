Getting Started
==================
stormtrooper is a lightweight Python library for zero and few-shot classification using transformer models.
All components are fully scikit-learn compatible thereby making it easier to integrate them into your scikit-learn workflows and pipelines.

Installation
^^^^^^^^^^^^
You can install stormtrooper from PyPI.

.. code-block::

   pip install stormtrooper[torch]

If you intend to use SetFit models as well, install stormtrooper with optional dependencies.

.. code-block::

   pip install stormtrooper[setfit]

From version 0.4.0 you can also use OpenAI models in stormtrooper.

.. code-block::

   pip install stormtrooper[openai]
   export OPENAI_API_KEY="sk-..."


Usage
^^^^^^^^^

To get started load a model from HuggingFace Hub.
In this example I am going to use Google's FLAN-T5.

.. code-block:: python

   from stormtrooper import Text2TextZeroShotClassifier

   class_labels = ["atheism/christianity", "astronomy/space"]
   example_texts = [
       "God came down to earth to save us.",
       "A new nebula was recently discovered in the proximity of the Oort cloud."
   ]

   model = Text2TextZeroShotClassifier("google/flan-t5-base").fit(None, class_labels)
   predictions = model.predict(example_texts)


.. toctree::
   :maxdepth: 1
   :caption: User guide

   zeroshot
   text2text
   generative
   setfit
   openai
   textgen
   prompting
   inference_on_gpu


.. toctree::
   GitHub Repository <https://github.com/centre-for-humanities-computing/stormtrooper>

Zero-shot models
================

In stormtrooper you can use models specifically designed and tuned for zero-shot classification.
These provide the most extensive functionality including estimating certainties/probabilities for class labels.


.. code-block:: python

   from stormtrooper import ZeroShotClassifier

   sample_text = "It is the Electoral College's responsibility to elect the president."

   model = ZeroShotClassifier("facebook/bart-large-mnli").fit(None, ["politics", "science", "other"])
   predictions = model.predict([sample_text])
   assert list(predictions) == ["politics"]
   
   model.set_output(transform="pandas")
   model.transform([sample_text])


+-----------+----------+---------+
|  politics |  science |  other  |
+===========+==========+=========+
|  0.924671 | 0.006629 |  0.0687 |
+-----------+----------+---------+

API reference
^^^^^^^^^^^^^

.. autoclass:: stormtrooper.ZeroShotClassifier
   :members:

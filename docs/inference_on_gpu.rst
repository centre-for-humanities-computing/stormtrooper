Inference on GPU
================

Models in stormtrooper run on the CPU by default, but you can influence this behaviour by specifying a device when initializing a model.

Make sure to check whether you have a Cuda device available before trying to run inference on it.

.. code-block:: python

   import torch

   print(torch.cuda.is_available())

.. code-block:: python

   from stormtrooper import Text2TextZeroShotClassifier
   
   model = Text2TextZeroShotClassifier(device="cuda:0")


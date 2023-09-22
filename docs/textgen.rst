Text Generation Inference
=========================

Text Generation Inference is a relatively new tool developed by Hugging Face for blazing fast deployment of generative and seq2seq models
for inference.
TGI provides numerous benefits over just running models in Transformers, including more efficient transformer layers, quantization,
and the ability to run LLM inference on a separate computer from where the machine learning pipeline is run.

stormtrooper comes with built-in support for connecting to a running instance of the TGI API and is the recommended solution
for production-ready pipelines, especially if you have access to CUDA GPUs and you are using a model that is natively supported in TGI.
stormtrooper also uses async requests for interacting with the TGI instance, as such inference speed is the only bottleneck in the pipeline.

.. image:: https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/TGI.png

Getting Started
^^^^^^^^^^^^^^^

We recommend that you follow the guide provided in HuggingFace's `text-generation-inference documentation <https://huggingface.co/docs/text-generation-inference/index>`_

In this tutorial I will demonstrate how to spin up StableBeluga-13b with TGI and then interface it from within stormtrooper.

By far the fastest way to get started is to start TGI in a Docker container. We will use all GPUs on the machine and use port 8080 on localhost,
which is the default in stormtrooper.

.. note::

   You need to install `NVIDIA's Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_ and CUDA drivers to be able to use GPU's in a Docker container.


.. code-block:: bash

   docker run --gpus all --shm-size 1g -p 8080:80 \
     -v $PWD/huggingface_data:/data ghcr.io/huggingface/text-generation-inference:1.0.3 \
     --model-id "stabilityai/StableBeluga-13B"


Then you can use TGI in stormtrooper as any other classification model.


.. code-block:: python

   from stormtrooper import TGIZeroShotClassifier, TGIFewShotClassifier

   sample_text = "It is the Electoral College's responsibility to elect the president."

   labels = ["politics", "science", "other"]


   model = TGIZeroShotClassifier().fit(None, labels)
   predictions = model.predict([sample_text])
   assert list(predictions) == ["politics"]
   
Gotchas
^^^^^^^

TGI has almost the same behaviour as any other module in stormtrooper but the interface differs subtlely, but crucially in a handful of instances.

1. You can use both text2text and purely generative models with TGI, but stormtrooper is entirely agnostic to which model is being run. As such you have to accomodate for these differences in your prompts. The default prompt format is suited for StableBeluga, but may be inappropriate for Google's FLAN-T5.
2. Most models in stormtrooper just slice the list of tokens in the prompt when it is too long. In contrast, with TGI you have to pay special attention to the length of your prompts. If the prompt is too long with the data infused, a warning will be raised and None will be returned. This behaviour may be subject to change in future releases.


API reference
^^^^^^^^^^^^^

.. autoclass:: stormtrooper.TGIZeroShotClassifier
   :members:

.. autoclass:: stormtrooper.TGIFewShotClassifier
   :members:

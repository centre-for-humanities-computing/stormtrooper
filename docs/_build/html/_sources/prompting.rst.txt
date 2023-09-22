Prompting
=========

Text2Text, Generative, TGI and OpenAI models use a prompting approach for classification.
stormtrooper comes with default prompts, but these might not suit the model you want to use,
or your use case might require a different prompting strategy from the default.
stormtrooper allows you to specify custom prompts in these cases.

Templates
^^^^^^^^^

Prompting in stormtrooper uses a templating approach, where the .format() method is called on prompts to
insert labels and data.

A zero-shot prompt for an instruct Llama model like Stable Beluga or for ChatGPT would look something like this (this is the default):

.. code-block:: python
  
   prompt = """
   ### System:
   You are a classification model that is really good at following
   instructions and produces brief answers
   that users can use as data right away.
   Please follow the user's instructions as precisely as you can.
   ### User:
   Your task will be to classify a text document into one
   of the following classes: {classes}.
   Please respond with a single label that you think fits
   the document best.
   Classify the following piece of text:
   '{X}'
   ### Assistant:
   """

   model = GenerativeZeroShotClassifier("stabilityai/StableBeluga-13B", prompt=prompt)

X represents the current text in question, while classes represents the classes learned from the data.

A few-shot prompt for let's say a T5 model looks as follows:

.. code-block:: python
  
   fewshot_prompt = """
   I will give you a piece of text. Please classify it
   as one of these classes: {classes}. Please only respond
   with the class label in the same format as provided.
   Here are some examples of texts labelled by experts.
   {examples}
   Label this piece of text:
   '{X}'
   """

You can at any stage emit any of the template variables from your prompts if you want to operate with definitions for example.


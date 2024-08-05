Prompting
=========

Text2Text, Generative, and OpenAI models use a prompting approach for classification.
stormtrooper comes with default prompts, but these might not be the best for your model or use case.
stormtrooper allows you to specify custom prompts in these cases.

Templates
^^^^^^^^^

Prompting in stormtrooper uses a templating approach, where the .format() method is called on prompts to
insert labels and data.

You can specify both user and system prompts for Generative and OpenAI models:

.. code-block:: python
  
   system_prompt = """
   You are a classification model that is really good at following
   instructions and produces brief answers
   that users can use as data right away.
   Please follow the user's instructions as precisely as you can.
   """

   user_prompt = """
   Your task will be to classify a text document into one
   of the following classes: {classes}.
   Please respond with a single label that you think fits
   the document best.
   Classify the following piece of text:
   '{X}'
   """

   model = GenerativeZeroShotClassifier("stabilityai/StableBeluga-13B", prompt=prompt, system_prompt=system_prompt)

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


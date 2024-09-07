import pytest

from stormtrooper import Trooper

train_texts = ["I love foxes", "I fucking hate foxes"]
train_labels = ["lovely", "terrible"]
test_texts = ["Rebecca is awesome!", "Wolves bad."]

models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "google/flan-t5-small",
    "facebook/bart-large-mnli",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]


def is_valid(texts, predictions: list[str]) -> bool:
    if not (len(predictions) == len(texts)):
        return False
    for prediction in predictions:
        if prediction not in train_labels:
            return False
    else:
        return True


@pytest.mark.parametrize("model", models)
def test_integration(model: str):
    trooper = Trooper(model)
    # Testing zero shot
    trooper.fit(None, train_labels)
    pred = trooper.predict(test_texts)
    assert is_valid(test_texts, pred)
    # testin few-shot
    trooper.fit(train_texts, train_labels)
    pred = trooper.predict(test_texts)
    assert is_valid(test_texts, pred)

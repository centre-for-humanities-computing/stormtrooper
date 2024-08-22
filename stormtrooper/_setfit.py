from typing import Iterable
import itertools
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    losses,
)
from datasets import Dataset


def generate_positive_pairs(texts, labels) -> Iterable[tuple[str, str]]:
    anchor = []
    positive = []
    label_to_texts = {label: [] for label in set(labels)}
    for text, label in zip(texts, labels):
        label_to_texts[label].append(text)
    for _, texts in label_to_texts.items():
        for anchor, positive in itertools.combinations_with_replacement(texts, 2):
            yield anchor, positive


def generate_synthetic_samples(
    y: Iterable[str],
    template_sentence: str = "This sentence is {label}",
    n_sample_per_label: int = 8,
) -> tuple[Iterable[str], Iterable[str]]:
    texts = []
    labels = []
    for label in set(y):
        text = template_sentence.format(label=label)
        for i in range(n_sample_per_label):
            texts.append(text)
            labels.append(label)
    return texts, labels


def finetune_contrastive(
    model: SentenceTransformer,
    texts: Iterable[str],
    labels: Iterable[str],
    n_epochs: int = 10,
    seed: int = 0,
) -> SentenceTransformer:
    pairs = generate_positive_pairs(texts, labels)
    anchor, positive = zip(*pairs)
    train_dataset = Dataset.from_dict(
        {"anchor": list(anchor), "positive": list(positive)}
    )
    train_dataset = train_dataset.shuffle(seed)
    loss = losses.MultipleNegativesRankingLoss(model)
    trainer = SentenceTransformerTrainer(
        model=model, train_dataset=train_dataset, loss=loss
    )
    trainer.train()
    return model

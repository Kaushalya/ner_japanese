from typing import Callable
import numpy as np
import seqeval


IGNORE_LABEL = -100

entity_map_en = {
    "法人名": "CORP",
    "その他の組織名": "ORG-O",
    "人名": "PER",
    "製品名": "PROD",
    "地名": "Place",
    "政治的組織名": "ORG-P",
    "施設名": "FAC",
    "イベント名": "EVT",
}

entity_names_en = list(entity_map_en.values())

# 0 is the label for the "O" tag
label_list = ["O"] + [
    f"{prefix}-{entity_name}"
    for entity_name in entity_names_en
    for prefix in ["B", "I"]
]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

label2id, id2label


def tokenize_and_align_labels(tokenizer, label2id: dict, entity_map: dict) -> Callable:
    """
    Align the NER labels with the tokenized inputs
    """

    def fn(examples):
        tokenized_inputs = tokenizer(examples["text"], return_offsets_mapping=True)
        labels = [label2id["O"]] * len(tokenized_inputs["input_ids"])

        for entity in examples["entities"]:
            entity_start, entity_end = entity["span"]
            label = entity_map[entity["type"]]
            # print(f"Entity: {text[entity_start:entity_end]}, Type: {label}")
            for i, (start, end) in enumerate(tokenized_inputs["offset_mapping"]):
                if start >= entity_start and end <= entity_end:
                    # Set the label of special tokens to -100
                    if start == end:
                        labels[i] = IGNORE_LABEL
                    # print(f"{i}/{len(labels)}")
                    elif start == entity_start:
                        labels[i] = label2id[f"B-{label}"]
                    else:
                        # Add I- prefix to the labels
                        labels[i] = label2id[f"I-{label}"]
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    return fn


def compute_metrics(p, label_list):
    """
    Compute metrics for the NER task
    """
    predictions, labels = p

    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != IGNORE_LABEL]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != IGNORE_LABEL]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

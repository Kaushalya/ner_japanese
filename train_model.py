import torch
import transformers
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    AutoModelForTokenClassification,
)
from transformers import TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset, get_dataset_split_names
import datasets
import numpy as np
import os
import evaluate
import yaml

# Disable NCCL to train on RT4000 series GPUs
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


def tokenize_and_align_labels(examples):
    """
    Align the NER labels with the tokenized inputs
    """
    tokenized_inputs = tokenizer(examples["text"], return_offsets_mapping=True)
    labels = [label2id["O"]] * len(tokenized_inputs["input_ids"])

    for entity in examples["entities"]:
        entity_start, entity_end = entity["span"]
        label = entity_map_en[entity["type"]]
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


def compute_metrics(p):
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load these from a config file

dataset_name = "stockmark/ner-wikipedia-dataset"
load_from_disk = True
dataset_path = "./data/ner-wikipedia-dataset"
learning_rate = 1e-5
seed = 123
save_steps = 1000
save_strategy = "steps"

if not load_from_disk:
    dataset = load_dataset(dataset_name)
    dataset.save_to_disk(dataset_path)
else:
    dataset = datasets.load_from_disk(dataset_path)

# Split the dataset into train and test. Train should be 80% of the data.
data_split = dataset["train"].train_test_split(train_size=0.8, seed=seed)
train_data, test_data = data_split["train"], data_split["test"]

enitity_names_jp = {e["type"] for ents in train_data["entities"] for e in ents}
print("Number of entity types:", len(enitity_names_jp))
print("Entity types:", enitity_names_jp)

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

IGNORE_LABEL = -100
# 0 is the label for the "O" tag
label_list = ["O"] + [
    f"{prefix}-{entity_name}"
    for entity_name in entity_names_en
    for prefix in ["B", "I"]
]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}


model_name = "xlm-roberta-large"
# model_name = "rinna/japanese-roberta-base"
# model_name = "tohoku-nlp/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenized_trainset = train_data.map(tokenize_and_align_labels)
tokenized_testset = test_data.map(tokenize_and_align_labels)

data_collator = DataCollatorForTokenClassification(tokenizer)

seqeval = evaluate.load("seqeval")

model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id
)

output_dir = f"./models/{model_name}"
log_dir = f"./runs/{model_name}"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=learning_rate,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=100,
    weight_decay=0.01,
    logging_steps=100,
    eval_strategy=save_strategy,
    save_strategy=save_strategy,
    save_steps=save_steps,
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to=["tensorboard"],
    logging_dir=log_dir,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_trainset,
    eval_dataset=tokenized_testset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()


def run_training():
    pass


# if __name__ == "__main__":
#     # Load a yaml file with the training parameters
#     with open("training_config.yaml") as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)
#     run_training(**config)

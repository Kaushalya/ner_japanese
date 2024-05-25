import argparse
import torch
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    AutoModelForTokenClassification,
)
from transformers import TrainingArguments
import datasets
import numpy as np
import os
import evaluate
import yaml
import argparse
from utils import (
    IGNORE_LABEL,
    entity_map_en,
    label2id,
    id2label,
    label_list,
    tokenize_and_align_labels,
)

# Disable NCCL to train on RT4000 series GPUs with Accelerate
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
seqeval = evaluate.load("seqeval")


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


def run_training(
    dataset_path: str,
    model_name: str,
    epochs: int,
    learning_rate: float,
    train_batch_size: int,
    eval_batch_size: int,
    seed: int,
    save_strategy: str = "epoch",
    save_total_limit: int = 5,
    save_steps: int = 1000,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_data = datasets.load_from_disk(os.path.join(dataset_path, "train"))
    test_data = datasets.load_from_disk(os.path.join(dataset_path, "test"))

    enitity_names_jp = {e["type"] for ents in train_data["entities"] for e in ents}
    print("Number of entity types:", len(enitity_names_jp))
    print("Entity types:", enitity_names_jp)

    # Tokenize the inputs and align the labels
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenize_fn = tokenize_and_align_labels(tokenizer, label2id, entity_map_en)
    tokenized_trainset = train_data.map(tokenize_fn)
    tokenized_testset = test_data.map(tokenize_fn)
    data_collator = DataCollatorForTokenClassification(tokenizer)

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
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy=save_strategy,
        save_strategy=save_strategy,
        save_steps=save_steps,
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to=["tensorboard"],
        logging_dir=log_dir,
        save_total_limit=save_total_limit,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training config file",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Training config could not be found at {args.config}")

    # Load a yaml file with the training parameters
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    run_training(**config)

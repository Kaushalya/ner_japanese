

import argparse

import datasets
import torch
from seqeval.metrics import classification_report
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from utils import (
    IGNORE_LABEL,
    entity_map_en,
    id2label,
    label2id,
    label_list,
    tokenize_and_align_labels,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    return parser.parse_args()


def main(dataset_dir, model_path):
    test_data = datasets.load_from_disk(dataset_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(
        model_path, num_labels=len(label_list), id2label=id2label, label2id=label2id
    ).to(device)
    model.eval()
    tokenize_fn = tokenize_and_align_labels(tokenizer, label2id, entity_map_en)
    tokenized_testset = test_data.map(tokenize_fn, batched=False)

    all_preds = []
    gt_labels = []

    print(f"Evaluating model {model_path} on test set")

    model.eval()
    for data in tqdm(tokenized_testset):
        inputs = {
            "input_ids": torch.tensor(data["input_ids"]).unsqueeze(0).to(device),
            "attention_mask": torch.tensor(data["attention_mask"])
            .unsqueeze(0)
            .to(device),
        }
        with torch.no_grad():
            preds = model(**inputs).logits[0]
        pred_classes = preds.argmax(dim=-1).cpu().tolist()
        pred_classes = [
            id2label[p]
            for p, l in zip(pred_classes, data["labels"])
            if l != IGNORE_LABEL
        ]
        gt_classes = [id2label[l] for l in data["labels"] if l != IGNORE_LABEL]
        all_preds.append(pred_classes)
        gt_labels.append(gt_classes)
    print(classification_report(gt_labels, all_preds))


if __name__ == "__main__":
    args = parse_args()
    main(args.dataset_dir, args.model_path)

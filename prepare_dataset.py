import argparse

from datasets import load_dataset
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="A dataset name from the Hugging Face datasets library",
        default="stockmark/ner-wikipedia-dataset",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to save the dataset",
        default="./data",
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.8,
        help="Fraction of the dataset to use for training",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    seed = args.seed
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    train_size = args.train_size

    dataset = load_dataset(dataset_name)

    data_split = dataset["train"].train_test_split(train_size=train_size, seed=seed)
    train_data, test_data = data_split["train"], data_split["test"]
    train_data.save_to_disk(os.path.join(dataset_path, "train"))
    test_data.save_to_disk(os.path.join("test"))

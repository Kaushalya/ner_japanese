from datasets import load_dataset
import yaml


if __name__ == "__main__":
    # Load variables from yaml config file
    with open("./configs/data_configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    dataset_name = config["dataset_name"]
    dataset_path = config["dataset_path"]
    train_size = config.get("train_size", 0.8)

    dataset = load_dataset(dataset_name)

    data_split = dataset["train"].train_test_split(train_size=train_size, seed=seed)
    train_data, test_data = data_split["train"], data_split["test"]
    train_data.save_to_disk(f"{dataset_path}/train")
    test_data.save_to_disk(f"{dataset_path}/test")

    data_split = dataset["train"].train_test_split(train_size=0.8, seed=seed)
    train_data, test_data = data_split["train"], data_split["test"]

    train_data.save_to_disk(f"{dataset_path}/train")
    test_data.save_to_disk(f"{dataset_path}/test")

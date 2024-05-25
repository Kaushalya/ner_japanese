
# Japanese Named Entity Recognition (NER) Model
## Download and split the dataset
You can download a Hugging Face dataset and split it into train and test splits using the `prepare_dataset.py` script. For example, the following command downloads the Japanese NER dataset `stockmark/ner-wikipedia-dataset` and splits creates a training set using 80% examples.
```
python prepare_dataset.py --dataset_name stockmark/ner-wikipedia-dataset --dataset_path results/data --seed 1234 --train_size 0.8
```

## Fine-tuned Models
Models finetuned using the `stockmark/ner-wikipedia-dataset` dataset is included in the `./models` folder.

## Evaluation
You can evaluate a pretrained model on a dataset stored at `dataset_dir` using the following script. A hugging face model name can also be used in place of `dataset_dir`.
```
python evaluate_model.py --dataset_dir ./data/ner-wikipedia-dataset/test --model_path  "./models/xlm-roberta-large/checkpoint-13000"
```
This command will print the performance of the model as a table.

### Performance of the `xlm-roberta-base` model
| Category     | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| CORP         | 0.97      | 0.97   | 0.97     | 547     |
| EVT          | 0.96      | 0.97   | 0.96     | 201     |
| FAC          | 0.98      | 0.97   | 0.98     | 238     |
| ORG-O        | 0.93      | 0.96   | 0.95     | 191     |
| ORG-P        | 0.95      | 0.96   | 0.96     | 302     |
| PER          | 0.98      | 0.98   | 0.98     | 683     |
| PROD         | 0.98      | 0.98   | 0.98     | 245     |
| Place        | 0.95      | 0.98   | 0.96     | 391     |
| **micro avg** | **0.97**  | **0.97** | **0.97** | **2798** |
| **macro avg** | **0.96**  | **0.97** | **0.97** | **2798** |
| **weighted avg** | **0.97** | **0.97** | **0.97** | **2798** |


### Performance of the `xlm-roberta-large` model
| Category     | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| CORP         | 0.97      | 0.97   | 0.97     | 547     |
| EVT          | 0.98      | 1.00   | 0.99     | 201     |
| FAC          | 0.97      | 0.99   | 0.98     | 238     |
| ORG-O        | 0.96      | 0.97   | 0.97     | 191     |
| ORG-P        | 0.96      | 0.97   | 0.97     | 302     |
| PER          | 0.99      | 0.98   | 0.98     | 683     |
| PROD         | 0.97      | 0.98   | 0.97     | 245     |
| Place        | 0.96      | 0.96   | 0.96     | 391     |
| **micro avg** | **0.97**  | **0.98** | **0.97** | **2798** |
| **macro avg** | **0.97**  | **0.98** | **0.97** | **2798** |
| **weighted avg** | **0.97** | **0.98** | **0.97** | **2798** |

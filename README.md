
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

## Performance
`xlm-roberta-base` and `xlm-roberta-large` models were trained for 100 epochs on two NVIDIA RTX 9040 GPUs.

### Performance of the `xlm-roberta-base` model
|                | precision | recall | f1-score | support |
|----------------|------------|--------|----------|---------|
| CORP           | 0.88       | 0.87   | 0.88     | 547     |
| EVT            | 0.85       | 0.88   | 0.86     | 201     |
| FAC            | 0.82       | 0.87   | 0.84     | 238     |
| ORG-O          | 0.72       | 0.76   | 0.74     | 191     |
| ORG-P          | 0.81       | 0.86   | 0.83     | 302     |
| PER            | 0.91       | 0.93   | 0.92     | 683     |
| PROD           | 0.75       | 0.81   | 0.78     | 245     |
| Place          | 0.86       | 0.91   | 0.88     | 391     |
| **micro avg**  | 0.85       | 0.88   | 0.86     | 2798    |
| **macro avg**  | 0.82       | 0.86   | 0.84     | 2798    |
| **weighted avg** | 0.85       | 0.88   | 0.86     | 2798    |


### Performance of the `xlm-roberta-large` model
| Category     | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| CORP         | 0.91      | 0.90   | 0.91     | 547     |
| EVT          | 0.91      | 0.94   | 0.92     | 201     |
| FAC          | 0.86      | 0.92   | 0.89     | 238     |
| ORG-O        | 0.82      | 0.86   | 0.84     | 191     |
| ORG-P        | 0.87      | 0.89   | 0.88     | 302     |
| PER          | 0.95      | 0.96   | 0.95     | 683     |
| PROD         | 0.80      | 0.82   | 0.81     | 245     |
| Place        | 0.89      | 0.92   | 0.90     | 391     |
| **micro avg**| 0.89      | 0.91   | 0.90     | 2798    |
| **macro avg**| 0.88      | 0.90   | 0.89     | 2798    |
| **weighted avg** | 0.89  | 0.91   | 0.90     | 2798    |

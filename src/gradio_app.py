from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import gradio as gr
import argparse

# model_name = "../models/xlm-roberta-base/checkpoint-3400"

label2id = {
    "O": 0,
    "B-CORP": 1,
    "I-CORP": 2,
    "B-ORG-O": 3,
    "I-ORG-O": 4,
    "B-PER": 5,
    "I-PER": 6,
    "B-PROD": 7,
    "I-PROD": 8,
    "B-Place": 9,
    "I-Place": 10,
    "B-ORG-P": 11,
    "I-ORG-P": 12,
    "B-FAC": 13,
    "I-FAC": 14,
    "B-EVT": 15,
    "I-EVT": 16,
}
id2label = {v: k for k, v in label2id.items()}

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
args = parser.parse_args()
model_name = args.model_path


def load_model():
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ner_model = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        ignore_labels=[label2id["O"]],
        device=0,
    )
    return ner_model


if gr.NO_RELOAD:
    ner_model = load_model()


# Function to get NER tags
def get_ner_tags(text):
    # tokens = tokenizer.tokenize(text)
    # inputs = tokenizer.encode(text, return_tensors="pt")
    # outputs = model(inputs).logits
    predictions = ner_model(text)

    # Combine nearby B- and I- labels
    prev_label = ""
    word = ""
    pred_entities = []

    entity_start = 0
    entity_end = 0

    for pred in predictions:
        if pred["start"] == pred["end"]:
            continue
        if pred["entity"][:2] != "I-":
            if prev_label != "":
                entity = {
                    "start": entity_start,
                    "end": entity_end,
                    "entity": prev_label,
                    "word": word,
                }
                pred_entities.append(entity)
                print(entity)
        if pred["entity"][:2] == "B-":
            entity_start = pred["start"]
            prev_label = pred["entity"][2:]
            word = pred["word"]
        elif pred["entity"][:2] == "I-":
            entity_end = pred["end"]
            word += pred["word"]
        else:
            prev_label = ""
            word = ""
    return {"text": text, "entities": pred_entities}


# Create the Gradio interface
interface = gr.Interface(
    fn=get_ner_tags,
    inputs=gr.Textbox(
        label="Input",
        value="",
    ),
    outputs=gr.HighlightedText(),
    title="Named Entity Recognition",
)

# Launch the Gradio interface
interface.launch(share=False)

from pyexpat import model
import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import numpy as np

# Load the fine-tuned model and tokenizer
model_name = "./models/xlm-roberta-base/checkpoint-3400"  # Adjust to the path where your model is saved
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


@st.cache_resource
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


# Define label mapping
# label_list = [
#     "O",
#     "人名",
#     "政治的組織名",
#     "GPE",
#     "I-ORG",
# ]
label_list_en = [
    "O",
    "PER",
    "ORG-P",
    "GPE",
    "Prod",
    "Place",
    "Fac",
    "EVT",
]

label_colors = [
    "Black",
    "Red",
    "Green",
    "Blue",
    "#FFA500",
    "#FFFF00",
    "#00FF00",
    "#0000FF",
    "#800080",
]
color_map = {
    label: color for label, color in zip(label_list_en, label_colors[: len(label_list)])
}

model = load_model()


# Function to get NER tags
def get_ner_tags(text):
    # tokens = tokenizer.tokenize(text)
    # inputs = tokenizer.encode(text, return_tensors="pt")
    # outputs = model(inputs).logits
    predictions = model(text)

    # Combine nearby B- and I- labels
    prev_label = ""
    word = ""
    pred_entities = []

    entity_start = 0
    entity_end = 0

    for pred in predictions:
        if pred["entity"][:2] != "I-":
            if prev_label != "":
                entity = {
                    "start": entity_start,
                    "end": entity_end,
                    "type": prev_label,
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
    return pred_entities


# Streamlit UI
st.title("Japanese Named Entity Recognition (NER)")
# Initialize session state for the text area
if "input_area" not in st.session_state:
    st.session_state["input_area"] = ""

sentence = st.text_area(
    "Enter a Japanese sentence:", st.session_state["input_area"]
)

if st.button("Submit"):
    if sentence:
        ner_tags = get_ner_tags(sentence)
        st.write(ner_tags)

        # Display results
        st.write("### Output")
        output_text = " ".join(
            [
                f"<span style='color:{color_map.get(tag['type'], 'black')}'>{tag['word']}[{tag['type']}]</span>"
                if tag["type"] != "O"
                else tag["word"]
                for tag in ner_tags
            ]
        )
        st.write(
            output_text,
            unsafe_allow_html=True,
        )
        # Update session state
        # st.session_state['text_area'] = sentence
    else:
        st.write("Please enter a sentence.")
if st.button("Clear"):
    st.write("Cleared the input.")
    st.session_state["input_area"] = ""
    st.rerun()

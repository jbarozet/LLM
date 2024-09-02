# Description:
#   Hugging Face transformers library
#   Text classification
#
# Last updated: 2024-02-26
#

from transformers import pipeline

# --- [ TEXT ]

text = """
Hugging Face is an AI company that has become a major hub for open-source machine learning.
They have 3 major elements which allow users to access and share machine learning resources.
"""

# --- [ PRINT OUTPUT ]

def print_output(number, text):
    print(f"\n\nExample {number}")
    for item in text:
        label = item["label"]
        score = item["score"]
        print(f"Label: {label} - Score:", score)


# --- [ Example 1 - Summarization with default model]
# The pipeline loads a default model automatically
# (here, the model loaded is distilbert-base-uncased-finetuned-sst-2-english)

pipe = pipeline("text-classification")
output = pipe("This restaurant is awesome")
print(output)

# --- [ Example 2 - Summarization with a model that has multiple target labels ]

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
output = classifier(text)
print_output(2, output[0])

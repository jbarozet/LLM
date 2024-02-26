# Description:
#   Hugging Face transformers library
#   Sentiment analysis
#   automated process of tagging data according to their sentiment, 
#   such as positive, negative and neutral.
#
# Last updated: 2024-02-26
#

from transformers import pipeline

# --- [ PRINT OUTPUT ]

def print_output(number, text):
    print(f"\n\nExample {number}")
    for item in text:
        label = item["label"]
        score = item["score"]
        print(f"Label: {label} - Score:", score)


# --- [ example 1 ]

output = pipeline(task="sentiment-analysis")("Love this!")
print_output(1, output)

# --- [ example 2 ]

output = pipeline(task="sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")("Love this!")
print_output(2, output)

# --- [ Example 3 - defining classifier ]

classifier = pipeline(task="sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
output = classifier("Hate this.")
print_output(3, output)

# --- [ Example 4 - we can also pass in a list to classifier ]

text_list = [
    "This is great",
    "Thanks for nothing",
    "You've got to work on your face",
    "You're beautiful, never change!"]

output = classifier(text_list)
print_output(4, output)

# print(f"example4: {output} ")
# payloadJSON = json.dumps(output, indent=4)
# print(f"example4: {payloadJSON} ")

# --- [ Example 5 - if there are multiple target labels, we can return them all ]

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
output = classifier(text_list[0])
print_output(5, output[0])


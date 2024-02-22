import sys
import json
from transformers import pipeline, Conversation

# toy example 1
output = pipeline(task="sentiment-analysis")("Love this!")
print(f"example1: {output}")

# toy example 2
output = pipeline(task="sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")("Love this!")
print(f"example2: {output}")

# defining classifier
classifier = pipeline(task="sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
output = classifier("Hate this.")
print(f"example3: {output}")

# we can also pass in a list to classifier
text_list = ["This is great", \
             "Thanks for nothing", \
             "You've got to work on your face", \
             "You're beautiful, never change!"]

output = classifier(text_list)
payloadJSON = json.dumps(output, indent=4)
print(f"example4: {payloadJSON} ")

# if there are multiple target labels, we can return them all
classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
output = classifier(text_list[0])
payloadJSON = json.dumps(output, indent=4)
print(f"example5: {payloadJSON} ")


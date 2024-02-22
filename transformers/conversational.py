import sys
import json
from transformers import pipeline, Conversation

chatbot = pipeline(model="facebook/blenderbot-400M-distill")
conversation = chatbot("Hi I'm Shaw, how are you?")
print(conversation)

conversation = chatbot("Where do you work?")
print(conversation)


# Description:
#   This is a basic example of how to use
#   Hugging Face transformers library
#
# Requires:
#   - transformers
#   - pytorch: https://pytorch.org/get-started/locally/
#
# Last updated: 2024-02-26
#

from transformers import pipeline

chatbot = pipeline(model="facebook/blenderbot-400M-distill")
conversation = chatbot("Hi I'm John, how are you?")
print(conversation)

conversation = chatbot("Where do you work?")
print(conversation)

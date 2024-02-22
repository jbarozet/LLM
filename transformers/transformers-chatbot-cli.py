# ------------------------------------------------------------------------------
#  _     _     __  __
# | |   | |   |  \/  |
# | |   | |   | |\/| |
# | |___| |___| |  | |
# |_____|_____|_|  |_|
#
# Chatbot using Hugging Face transformers library
#
# Requires:
# - transformers
# - torch
#
# pip install transformers (Hugging Face library)
# pip install 'transformers[torch]' (deep learning library )
# ------------------------------------------------------------------------------

import sys
from transformers import pipeline, Conversation

chatbot = pipeline(model="facebook/blenderbot-400M-distill")

conversation = Conversation("Hi I'm JMB, how are you?")
conversation = chatbot(conversation)  
print(conversation) 


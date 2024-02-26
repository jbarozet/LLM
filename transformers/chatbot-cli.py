# DESCRIPTION
#   Chatbot using Hugging Face transformers library
#
# Last updated: 2024-02-26

from transformers import pipeline, Conversation
# from ctransformers AutoModelForCausalLM, AutoTokenizer

chatbot = pipeline(model="facebook/blenderbot-400M-distill")

conversation = Conversation("Hi I'm JMB, how are you?")
# conversation = chatbot(conversation)
print(conversation)

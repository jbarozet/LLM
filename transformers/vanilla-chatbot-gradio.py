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
# - gradio
# - transformers
# - torch
#
# pip install gradio (UI framework)
# pip install transformers (Hugging Face library)
# pip install 'transformers[torch]' (deep learning library )
#
#  Updated: jmb (2024-02)
#
# ------------------------------------------------------------------------------

from transformers import pipeline, Conversation
# from ctransformers AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModel
import gradio as gr

#chatbot = pipeline(model="facebook/blenderbot-400M-distill")
# model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-Chat-GGUF", hf=True)
# tokenizer = AutoTokenizer.from_pretrained(model)
# pipe = pipeline(model=model, tokenizer=tokenizer)
# print(pipe("AI is going to", max_new_tokens=256))

model = AutoModel.from_pretrained("TheBloke/CodeLlama-7B-Python-GGUF")
chatbot = pipeline(model=model)

message_list = []
response_list = []

def vanilla_chatbot(message, history):
    conversation = chatbot(message)
    return conversation[0]['generated_text']


demo_chatbot = gr.ChatInterface(vanilla_chatbot, title="Vanilla Chatbot", description="Enter text to start chatting.")

demo_chatbot.launch()


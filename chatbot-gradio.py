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
import gradio as gr

chatbot = pipeline(model="facebook/blenderbot-400M-distill")

message_list = []
response_list = []

def vanilla_chatbot(message, history):
    conversation = Conversation(text=message, past_user_inputs=message_list, generated_responses=response_list)
    conversation = chatbot(conversation)

    return conversation.generated_responses[-1]

demo_chatbot = gr.ChatInterface(vanilla_chatbot, title="Vanilla Chatbot", description="Enter text to start chatting.")

demo_chatbot.launch()

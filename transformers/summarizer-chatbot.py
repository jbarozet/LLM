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
# - gradio
#
# pip install transformers (Hugging Face library)
# pip install 'transformers[torch]' (deep learning library )
# ------------------------------------------------------------------------------
import sys
import json
from transformers import pipeline, Conversation
import gradio as gr

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarizer_bot(message, history):
    return summarizer(message, min_length=5, max_length=140)[0]['summary_text']


demo_summarizer = gr.ChatInterface(summarizer_bot, title="Summarizer Chatbot", description="Enter your text, and the chatbot will return the summarized version.")

demo_summarizer.launch()

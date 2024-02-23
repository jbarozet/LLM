# Description: run a chatbot using Ollama and gradio
# Usage: python chatbot1.py
# This will display a URL that you can use to chat
#
# Last Updated: 2024-02-22

from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import gradio as gr


# Conversation
conversation_history = []

# LLM parameters
temperature = 0.75
max_tokens = 2000
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Load a model using ollama

llm = Ollama(model="mistral", callback_manager=callback_manager)

print(f"Loaded LLM model {llm.model}")

def generate_response(prompt):
    conversation_history.append(prompt)
    full_prompt = "\n".join(conversation_history)
    output = llm.invoke(full_prompt)
    return output


# Load graphical interface
iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=2, placeholder="Enter your prompt here..."),
    outputs="text",
)

iface.launch()


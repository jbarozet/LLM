# Description:
#   run a chatbot using Ollama and gradio
#   using the requests library
#
# Usage:
#   python chatbot3.py
#   This will display a URL that you can use to chat
#
# Last Updated: 2024-09-02
import json

import gradio as gr
import requests

url = "http://localhost:11434/api/generate"

headers = {"Content-Type": "application/json"}

history = []


def generate_response(prompt):
    history.append(prompt)
    final_prompt = "\n".join(history)

    data = {"model": "mistral", "prompt": final_prompt, "stream": False}

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        response = response.text
        data = json.loads(response)
        actual_response = data["response"]
        return actual_response
    else:
        print("error", response.text)


interface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=4, placeholder="Enter your Promt"),
    outputs="text",
)
interface.launch()

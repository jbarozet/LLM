# ------------------------------------------------------------------------------
#  _     _     __  __
# | |   | |   |  \/  |
# | |   | |   | |\/| |
# | |___| |___| |  | |
# |_____|_____|_|  |_|
#
#  Updated: jmb (2024-02-26)
#
# DESCRIPTION
#   Very simple chatbot using llama-cpp
#   and a model downloaded from Hugging Face
#
# INSTALL
#   poetry add langchain-community
#   poetry add llama-cpp-python
#
# Just download a suitable pre-trained model.
# I’m using synthia-7b-v2 (or llama2) here with impressive results on general questions formulated in English.
# Also tried with llama2
#
# Hugging Face LLMs:
#   https://huggingface.co/TheBloke/SynthIA-7B-v2.0-16k-GGUF
#   https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
#
# ------------------------------------------------------------------------------

import sys
# from langchain.llms import LlamaCpp
from langchain_community.llms import LlamaCpp

# enable verbose to debug the LLM's operation
verbose = False

model_path = "/Users/jbarozet/LLM/llama-2-7b-chat.Q4_K_M.gguf"
# model_path="/Users/jbarozet/LLM/synthia-7b-v2.0-16k.Q4_K_M.gguf"

# With CPU
llm = LlamaCpp(
    model_path=model_path,
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    verbose=True,  # Verbose is required to pass to the callback manager
)

while True:
    question = input("Ask me a question: ")

    if question == "/bye":
        sys.exit(1)
    output = llm.invoke(question)

    print(f"\n{output}")

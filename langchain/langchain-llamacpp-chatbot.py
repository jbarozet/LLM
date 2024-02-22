# ------------------------------------------------------------------------------
#  _     _     __  __
# | |   | |   |  \/  |
# | |   | |   | |\/| |
# | |___| |___| |  | |
# |_____|_____|_|  |_|
#
#
#  Updated: jmb (2024-02)
#
# NOTES
# I just import two libraries: sys and langchain.llms.
# Importing LLMs from langchain is deprecated. Importing from langchain will no longer be supported as of langchain==0.2.0.
# Please import from langchain-community instead
# The LlamaCpp module itself requires the installation of dependencies (pip install llama-cpp-python).
#
# Not used anymore: pip install langchain
# pip install -U langchain-community
# pip install llama-cpp-python
#
# Just download a suitable pre-trained model.
# I’m using synthia-7b-v2 here with impressive results on general questions formulated in English.
#
# https://huggingface.co/TheBloke/SynthIA-7B-v2.0-16k-GGUF
#
# ------------------------------------------------------------------------------

import sys
# from langchain.llms import LlamaCpp
from langchain_community.llms import LlamaCpp

# enable verbose to debug the LLM's operation
verbose = False

model_path = "/Users/jbarozet/LLM/llama-2-7b-chat.Q4_K_M.gguf"
#model_path="/Users/jbarozet/LLM/synthia-7b-v2.0-16k.Q4_K_M.gguf"

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

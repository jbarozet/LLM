# ------------------------------------------------------------------------------
#
#   ___  _     _        _    __  __    _
#  / _ \| |   | |      / \  |  \/  |  / \
# | | | | |   | |     / _ \ | |\/| | / _ \
# | |_| | |___| |___ / ___ \| |  | |/ ___ \
#  \___/|_____|_____/_/   \_\_|  |_/_/   \_\
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

llm = LlamaCpp(
    model_path="/home/jmb/LLM/synthia-7b-v2.0-16k.Q4_K_M.gguf",
    # max tokens the model can account for when processing a response
    # make it large enough for the question and answer
    n_ctx=4096,
    # number of layers to offload to the GPU
    # GPU is not strictly required but it does help
    n_gpu_layers=32,
    # number of tokens in the prompt that are fed into the model at a time
    n_batch=1024,
    # use half precision for key/value cache; set to True per langchain doc
    f16_kv=True,
    verbose=verbose,
)

while True:
    question = input("Ask me a question: ")
    if question == "stop":
        sys.exit(1)
    output = llm(
        question,
        max_tokens=4096,
        temperature=0.2,
        # nucleus sampling (mass probability index)
        # controls the cumulative probability of the generated tokens
        # the higher top_p the more diversity in the output
        top_p=0.1
    )
    print(f"\n{output}")

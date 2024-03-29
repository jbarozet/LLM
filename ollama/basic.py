# Description:
#    This is a basic example of how to use OLLAMA as an LLM in Python
#    Using langchain library
#
# Usage:
#   Download and install ollama
#   Pull llama2: ollama pull llama2
#   run: python3 basic.py
#
# Last updated: 2024-02-26
#
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = Ollama(model="llama2", callback_manager=callback_manager)

llm.invoke("Tell me 5 facts about Roman history:")

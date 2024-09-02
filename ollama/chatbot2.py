# Description:
#   run a chatbot using a local model using (ollama) and streamlit (web framework)
#
# Usage:
#   streamlit run chatbot2.py
#
# Last Updated: 2024-09-02
#
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA

# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Conversation
conversation_history = []

# LLM parameters
# model_path = "/Users/jbarozet/LLM/llama-2-7b-chat.Q4_K_M.gguf"
# model_path="/Users/jbarozet/LLM/synthia-7b-v2.0-16k.Q4_K_M.gguf"

# Load a model using ollama
llm = Ollama(model="llama2")
print(f"Loaded LLM model {llm.model}")

colA, colB = st.columns([0.90, 0.10])

with colA:
    prompt = st.text_input("prompt", value="", key="prompt")
response = ""

with colB:
    st.markdown("")
    st.markdown("")
    if st.button("👉", key="button"):
        response = llm.invoke(prompt)

st.markdown(response)

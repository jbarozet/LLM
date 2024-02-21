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
# ------------------------------------------------------------------------------

from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Create a model using ollama
ollama = Ollama(base_url='http://localhost:11434', model="llama2")
# print(ollama("why is the sky blue"))

# Now let's load a document to ask questions against.
# I'll load up the Odyssey by Homer, which you can find at Project Gutenberg.
# We will need WebBaseLoader which is part of LangChain and loads text from any webpage.
# On my machine, I also needed to install bs4 to get that to work, so run pip install bs4.
loader = WebBaseLoader("https://www.gutenberg.org/files/1727/1727-h/1727-h.htm")
data = loader.load()

# This file is pretty big. Just the preface is 3000 tokens.
# Which means the full document won't fit into the context for the model.
# So we need to split it up into smaller pieces.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# It's split up, but we have to find the relevant splits and then submit those to the model.
# We can do this by creating embeddings and storing them in a vector database.
# We can use Ollama directly to instantiate an embedding model.
# We will use ChromaDB in this example for a vector database. pip install GPT4All chromadb
oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="llama2")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)

# Now let's ask a question
# This will output the number of matches for chunks of data similar to the search.
question = "Who is Neleus and who is in Neleus' family?"
docs = vectorstore.similarity_search(question)
len(docs)

# The next thing is to send the question and the relevant parts of the docs to the model to see if we can get a good answer.
# But we are stitching two parts of the process together, and that is called a chain.
# This means we need to define a chain:
qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
qachain({"query": question})


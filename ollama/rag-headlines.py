# Description: how to use Ollama with langchain and RAG.
# Get headlines from a news site.
#
# Usage:
# python3 rag-headlines --url https://techcrunch.com/
#
# Last Updated: 2024-02-24

# Load web page
import argparse

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embed and store
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings import OllamaEmbeddings # We can also try Ollama embeddings

from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# RAG Prompt
from langchain import hub

# QA Chain
from langchain.chains import RetrievalQA


def main():
    # Parse arguments and get url
    parser = argparse.ArgumentParser(description='Filter out URL argument.')
    parser.add_argument('--url', type=str, default='http://example.com', required=True, help='The URL to filter out.')

    args = parser.parse_args()
    url = args.url
    print(f"using URL: {url}")

    # Now let's load a document to ask questions against.
    # We will need WebBaseLoader which is part of LangChain and loads text from any webpage
    loader = WebBaseLoader(url)
    data = loader.load()
    print(f"Loaded {len(data)} documents")
    # print(f"Retrieved {len(docs)} documents")

    # Document could be pretty big.
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(data)
    print(f"Split into {len(all_splits)} chunks")

    # It's split up, but we have to find the relevant splits and then submit those to the model.
    # We can do this by creating embeddings and storing them in a vector database.
    # We can use Ollama directly to instantiate an embedding model.
    # We will use ChromaDB in this example for a vector database. pip install GPT4All chromad
    # embed = GPT4AllEmbeddings()
    oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="llama2")
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)

    # Now let's ask a question
    # This will output the number of matches for chunks of data similar to the search.
    question = "What are the latest headlines on {url}?"
    docs = vectorstore.similarity_search(question)
    len(docs)

    # LLM
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = Ollama(model="llama2", verbose=True, callback_manager=callback_manager)
    print(f"Loaded LLM model {llm.model}")

    # RAG prompt
    QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")

    # QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    # Ask a question
    question = f"What are the latest headlines on {url}?"
    result = qa_chain.invoke({"query": question})

    print(result)


if __name__ == "__main__":
    main()

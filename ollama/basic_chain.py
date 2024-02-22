# Description: This is a basic example of how to use OLLAMA as an LLM in Python and use it as a RAG
# run: python3 rag.py --url https://techcrunch.com/
# Last updated: 2024-02-22
#
from langchain_community.llms import Ollama
# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = Ollama(model="llama2", temperature=0.9)

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Give me 5 interesting facts about {topic}?",
)

chain = LLMChain(llm=llm, prompt=prompt, verbose=False)

# Run the chain only specifying the input variable.
output = chain.invoke("Paris")
print(output['text'])

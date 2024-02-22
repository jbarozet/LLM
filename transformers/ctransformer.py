import sys
from langchain_community.llms import CTransformers
import gradio as gr


llm = CTransformers(
    model="TheBloke/CodeLlama-7B-Python-GGUF",
    model_type="llama",
    max_new_tokens=2000,
    temperature=0.5,
)

while True:
    question = input("Ask me a question: ")

    if question == "/bye":
        sys.exit(1)

    output = llm.invoke(question)

    print(f"\n{output}")

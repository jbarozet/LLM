# Examples of LLM scripts

A collection of very simple examples to provide a ChatGPT-like interface, but running locally on your PC/Mac, and using ollama, langchains, transformers.

## Ollama

Start ollama server:

```console
ollama serve
```

Run ollama with mistral model:

```console
ollama run mistral
```

Then run the script, which will launch a web interface:

```console
python3 openchat.py
```

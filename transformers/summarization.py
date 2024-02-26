# Description:
#   Hugging Face transformers library
#   Summarizing text
#
# Last updated: 2024-02-26
#

from transformers import pipeline

# --- [ TEXT ]

text = """
Hugging Face is an AI company that has become a major hub for open-source machine learning. 
Their platform has 3 major elements which allow users to access and share machine learning resources. 
First, is their rapidly growing repository of pre-trained open-source machine learning models for things such as natural language processing (NLP), computer vision, and more. 
Second, is their library of datasets for training machine learning models for almost any task. 
Third, and finally, is Spaces which is a collection of open-source ML apps.

The power of these resources is that they are community generated, which leverages all the benefits of open source i.e. cost-free, wide diversity of tools, high quality resources, and rapid pace of innovation. 
While these make building powerful ML projects more accessible than before, there is another key element of the Hugging Face ecosystem—their Transformers library.
"""

# --- [ PRINT OUTPUT ]

def print_output(text):
    for item in text:
        label = item["label"]
        score = item["score"]
        print(f"Label: {label} - Score:", score)


# --- [ TEXT SUMMARIZATION ]

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

summarized_text = summarizer(text, min_length=5, max_length=140)[0]['summary_text']
print(f"Summarized Text: \n{summarized_text}\n\n")

# --- [ TEXT CLASSIFICATION ]

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
output = classifier(summarized_text)
text_classification = output[0]
print_output(text_classification)

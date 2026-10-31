# pip install langchain-text-splitters langchain-experimental langchain-openai

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings


text = """
# LangChain

LangChain is a framework for building applications with Large Language Models.

## Text Splitting

Text splitting divides large documents into smaller chunks.
These chunks can then be used for searching and processing.

## Embeddings

Embeddings convert text into numerical representations.

# Python

Python is a programming language.
It is commonly used for web development and data science.
"""


# --------------------------------------------------
# 1. LENGTH-BASED
# Size decides → number of characters
# --------------------------------------------------

splitter1 = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks1 = splitter1.split_text(text)

print("1. LENGTH BASED")
for chunk in chunks1:
    print(chunk)
    print("---")


# --------------------------------------------------
# 2. RECURSIVE CHARACTER SPLITTER
# Splits using separators such as:
# paragraphs → sentences → words → characters
# --------------------------------------------------

splitter2 = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks2 = splitter2.split_text(text)

print("\n2. RECURSIVE CHARACTER SPLITTER")
for chunk in chunks2:
    print(chunk)
    print("---")


# --------------------------------------------------
# 3. DOCUMENT STRUCTURE-BASED
# Structure such as Markdown headings decides chunks
# --------------------------------------------------

headers = [
    ("#", "Main Topic"),
    ("##", "Section")
]

splitter3 = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers
)

chunks3 = splitter3.split_text(text)

print("\n3. DOCUMENT STRUCTURE BASED")
for chunk in chunks3:
    print(chunk)
    print("---")


# --------------------------------------------------
# 4. SEMANTIC MEANING-BASED
# Meaning/topic changes decide chunks
# --------------------------------------------------

embeddings = OpenAIEmbeddings()

splitter4 = SemanticChunker(embeddings,breakpoint_threshold_amount=1, breakpoint_threshold_type="standard_deviation")
#standard deviation 

chunks4 = splitter4.split_text(text)

print("\n4. SEMANTIC MEANING BASED")
for chunk in chunks4:
    print(chunk)
    print("---")
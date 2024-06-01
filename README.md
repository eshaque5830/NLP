# Tokenization in NLP
Tokenization in Natural Language Processing (NLP) is the process of splitting text into smaller units, called tokens. These tokens can be words, subwords, characters, or other meaningful elements. Tokenization is a fundamental step in preprocessing text data for NLP tasks, as it transforms raw text into a structured format that can be easily analyzed and processed by machine learning models.

## Types of Tokenization
### i- Word Tokenization:

Splitting text into individual words.
Example: "Hello, world!" -> ["Hello", ",", "world", "!"]

### ii- Subword Tokenization:
Splitting text into subword units or morphemes, which can help in handling unknown words or variations of words.
Example: "unhappiness" -> ["un", "happiness"] or "playing" -> ["play", "ing"]

### iii- Character Tokenization:
Splitting text into individual characters.
Example: "Hello" -> ["H", "e", "l", "l", "o"]

## Why Tokenization is Important?
Facilitates Text Analysis: Breaking down text into smaller components makes it easier to analyze and manipulate.
Enables Feature Extraction: Tokenization is the first step in creating features for machine learning models.
Improves Model Performance: Proper tokenization can enhance the performance of NLP models by providing them with meaningful units of text.
Examples of Tokenization
Word Tokenization
#### Using NLTK (Natural Language Toolkit):

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

text = "Hello, world! How are you?"
tokens = word_tokenize(text)
print(tokens)
#### Output: ['Hello', ',', 'world', '!', 'How', 'are', 'you', '?']

import spacy

nlp = spacy.load("en_core_web_sm")
text = "Hello, world! How are you?"
doc = nlp(text)
tokens = [token.text for token in doc]
print(tokens)
#### Output: ['Hello', ',', 'world', '!', 'How', 'are', 'you', '?']

### Subword Tokenization
Using Byte Pair Encoding (BPE) from the Hugging Face Transformers library:
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "unhappiness"
tokens = tokenizer.tokenize(text)
print(tokens)
#### Output: ['un', '##happiness']

### Character Tokenization
Simple Python implementation:
text = "Hello"
tokens = list(text)
print(tokens)
#### Output: ['H', 'e', 'l', 'l', 'o']

## Tools and Libraries for Tokenization
NLTK (Natural Language Toolkit): Provides simple and flexible tokenization functions.
spaCy: Offers efficient and fast tokenization as part of its NLP pipeline.
Hugging Face Transformers: Includes tokenization for various pre-trained models, supporting subword tokenization techniques like BPE and WordPiece.
Gensim: Useful for word tokenization and topic modeling.

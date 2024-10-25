#import nltk
#nltk.download('punkt')
#nltk.download('punkt_tab')

import nltk
import os

# Custom NLTK data path
NLTK_DATA_PATH = './nltk_data'
if not os.path.exists(NLTK_DATA_PATH):
    os.makedirs(NLTK_DATA_PATH)
nltk.data.path.append(NLTK_DATA_PATH)

# Ensure 'punkt' and 'punkt_tab' data are downloaded
try:
    nltk.data.find('/punkt')
    nltk.data.find('/punkt_tab')
except LookupError:
    nltk.download('punkt', download_dir=NLTK_DATA_PATH)
    nltk.download('punkt_tab', download_dir=NLTK_DATA_PATH)

print("NLTK resources are set up and ready.")

import re
import PyPDF2
from nltk.tokenize import sent_tokenize


def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text


def filter_important_sentences(sentences, keywords=None):
    if keywords is None:
        keywords = ["introduction", "conclusion", "summary", "objective", "goal"]

    important_sentences = [sentence for sentence in sentences if
                           any(keyword in sentence.lower() for keyword in keywords)]
    return important_sentences if important_sentences else sentences  # Fallback if no keywords are found


def preprocess_pdf_text(pdf_text, header=None, footer=None, keywords=None):
    # Clean the text and normalize spaces around punctuation
    pdf_text = re.sub(r'\s+', ' ', pdf_text)
    pdf_text = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', pdf_text)

    # Remove headers and footers if specified
    if header:
        pdf_text = pdf_text.replace(header, "")
    if footer:
        pdf_text = pdf_text.replace(footer, "")

    # Remove tables and images
    pdf_text = re.sub(r'Table [\d]+.*|Figure [\d]+.*', '', pdf_text)

    # Sentence segmentation
    sentences = sent_tokenize(pdf_text)

    # Filter sentences based on keywords
    important_sentences = filter_important_sentences(sentences, keywords)
    return important_sentences

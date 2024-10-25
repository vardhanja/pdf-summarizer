import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Set environment variable to avoid parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Select device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"


# Function to load model and tokenizer based on model name
def load_model(model_name="facebook/bart-large-cnn"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        print(f"Loaded model '{model_name}' on {device}.")
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Error loading model '{model_name}': {str(e)}")


# Summarization function with adjustable parameters
def summarize_text(text, model, tokenizer, max_input_length=1024, max_summary_length=300, min_summary_length=100):
    try:
        # Split text into chunks if it exceeds max_input_length
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_length).to(device)

        # Generate a longer summary by increasing max_length and min_length
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_summary_length,
            min_length=min_summary_length,
            num_beams=6,  # Increased beams for more comprehensive summaries
            repetition_penalty=1.1,  # Discourages repeating phrases
            early_stopping=True
        )

        # Decode and return the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary

    except IndexError:
        raise RuntimeError("IndexError during summarization, possibly due to input size or model state.")

    except Exception as e:
        raise RuntimeError(f"Error during summarization: {str(e)}")


# Chunk Summarization Function

def chunk_summarize(text, model, tokenizer, max_chunk_length=1024, max_summary_length=300, min_summary_length=100):
    # Split the text into chunks
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    summaries = []

    for chunk in chunks:
        summary = summarize_text(chunk, model, tokenizer, max_input_length=max_chunk_length,
                                 max_summary_length=max_summary_length, min_summary_length=min_summary_length)
        summaries.append(summary)

    # Combine summaries of all chunks
    return " ".join(summaries)


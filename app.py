import streamlit as st
from pdf_utils import extract_text_from_pdf, preprocess_pdf_text
from model_utils import load_model, chunk_summarize


def main():
    st.title("PDF Summarizer with Model Selection")

    # Check if model and tokenizer are in session state, if not, initialize them
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer = None

    # Upload PDF file
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Model selection
    model_option = st.selectbox("Choose a summarization model", ["facebook/bart-large-cnn", "t5-small", "t5-base"])

    # Button to load the model
    if st.button("Load Model"):
        st.session_state.model, st.session_state.tokenizer = load_model(model_name=model_option)
        st.success(f"Model '{model_option}' loaded successfully!")

    # Button to generate summary
    submit = st.button("Generate Summary")

    if uploaded_file is not None and submit:
        if st.session_state.model is None or st.session_state.tokenizer is None:
            st.error("Please load the model before generating the summary.")
            return

        # Extract text from PDF
        with st.spinner('Extracting text from PDF...'):
            pdf_text = extract_text_from_pdf(uploaded_file)

        # Apply preprocessing to clean the text and segment sentences
        with st.spinner('Preprocessing text...'):
            preprocessed_sentences = preprocess_pdf_text(pdf_text)
            preprocessed_text = " ".join(preprocessed_sentences)

        # Display preprocessed text (optional sample for verification)
        st.subheader("Preprocessed PDF Text (Sample)")
        st.write(preprocessed_text[:2000] + "...")  # Display a sample of the preprocessed text

        # Generate summary from preprocessed text with error handling
        try:
            with st.spinner('Generating summary...'):
                summary = chunk_summarize(preprocessed_text, st.session_state.model, st.session_state.tokenizer)
            st.subheader("PDF Summary")
            st.write(summary)
        except RuntimeError as e:
            st.error(f"Error during summarization: {str(e)}")


if __name__ == '__main__':
    main()
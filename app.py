import streamlit as st
from transformers import pipeline

# Load the model and tokenizer
model_name = "kohendru/distilbert-base-uncased-amazon-sentiment-analysis"
nlp = pipeline("text-classification", model=model_name, tokenizer=model_name)

# Streamlit app
st.title("Sentiment Analysis with DistilBERT")

# User input
st.write("Enter reviews below for sentiment analysis:")
review_input = st.text_area("Enter your review:", height=200)

if st.button("Analyze Sentiment"):
    if review_input.strip():
        # Perform sentiment analysis
        results = nlp(review_input)
        sentiment = results[0]['label']
        confidence = results[0]['score']

        # Display results
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence:.4f}")
    else:
        st.write("Please enter a valid review for analysis.")
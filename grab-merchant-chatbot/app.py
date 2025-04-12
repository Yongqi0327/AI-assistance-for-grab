import pandas as pd
import google.generativeai as genai
import os
from transformers import pipeline
from utils.prompt_builder import build_prompt
from dotenv import load_dotenv
load_dotenv()

# Insert the API key

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load your merchant dataset
df_items = pd.read_csv("data/items.csv")
df_keywords = pd.read_csv("data/keywords.csv")
df_merchant = pd.read_csv("data/merchant.csv")
df_transaction_data = pd.read_csv("data/transaction_data.csv")
df_transaction_items = pd.read_csv("data/transaction_items.csv")

# Initialize the Gemini model
model_name = "gemini-1.5-flash"  # Or another available model
gemini_model = genai.GenerativeModel(model_name)

# Use Gemini to get insig  hts or answer questions about merchant data
try:
    response = gemini_model.generate_content(
        "What are some marketing ideas for a small Grab merchant?"
    )

    # Extract the response from Gemini
    generated_text = response.text

    # Print the generated response (insight or answer)
    print("Generated Insight from Gemini:", generated_text)

    # Now use Hugging Face's transformers to either classify or summarize the response
    # Example: Summarize the generated text
    summarizer = pipeline("summarization")
    summary = summarizer(generated_text, max_length=50, min_length=25, do_sample=False)

    # Print the summary of the response
    print("Summary of the Insight:", summary[0]['summary_text'])

    # Example: Classify the sentiment of the generated text
    classifier = pipeline("text-classification", model="bert-base-uncased")
    classification_result = classifier(generated_text)

    # Print the classification result
    print("Classification Result (Sentiment):", classification_result)

except Exception as e:
    print(f"An error occurred: {e}")
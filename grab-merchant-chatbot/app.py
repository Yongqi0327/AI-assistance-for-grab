import pandas as pd
import google.generativeai as genai
import os
from transformers import pipeline

# Directly set your API key here
GOOGLE_API_KEY = "your_api_key"

# Configure Gemini with the provided API key
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the Gemini model
model_name = "gemini-1.5-flash"
gemini_model = genai.GenerativeModel(model_name)

# Load datasets (optional, if used for future context)
df_items = pd.read_csv("items.csv")
df_keywords = pd.read_csv("keywords.csv")
df_merchant = pd.read_csv("merchant.csv")
df_transaction_data = pd.read_csv("transaction_data.csv")
df_transaction_items = pd.read_csv("transaction_items.csv")

# Initialize other models
summarizer = pipeline("summarization")
classifier = pipeline("text-classification", model="bert-base-uncased")

# Define your set of keywords
keywords = {
    'sales': ['sales', 'best-selling', 'top seller', 'quantity sold'],
    'promotion': ['promotion', 'discount', 'offer', 'coupon'],
    'marketing': ['marketing', 'strategy', 'advertising', 'promotion'],
    'customer': ['customer', 'feedback', 'reviews', 'satisfaction'],
}

# Function to process dataset and get the best-selling item


def get_best_selling_item():
    # Aggregating sales by item_id
    item_sales = df_transaction_data.groupby("item_id")["quantity_sold"].sum()

    # Identifying the best-selling item (highest total sales)
    best_selling_item_id = item_sales.idxmax()

    # Getting the item details from the items dataframe
    best_selling_item = df_items[df_items["item_id"]
                                 == best_selling_item_id].iloc[0]

    return best_selling_item["item_name"], item_sales.max()

# Function to check if any keywords are present in the user question


def match_keywords(user_question):
    matched_keywords = []
    for category, words in keywords.items():
        for word in words:
            if word.lower() in user_question.lower():
                matched_keywords.append(category)
                break  # Stop after the first match in each category
    return matched_keywords

# Function to ask Gemini for prediction or insights based on dataset and matched keywords


def ask_gemini_for_prediction(user_question):
    # Get the matched keywords from the user's question
    matched_categories = match_keywords(user_question)

    # Get best-selling item details
    item_name, sales = get_best_selling_item()

    # Formulate the question for Gemini based on matched keywords
    if 'sales' in matched_categories:
        user_question = f"Based on the best-selling item '{item_name}' with {sales} units sold, what sales strategy should a small Grab merchant implement to increase sales?"
    elif 'promotion' in matched_categories:
        user_question = f"What promotion or discount strategy would work best for a Grab merchant selling '{item_name}'?"
    elif 'marketing' in matched_categories:
        user_question = f"What marketing strategies can a small Grab merchant use to boost sales for '{item_name}'?"
    elif 'customer' in matched_categories:
        user_question = f"What customer feedback strategies can a Grab merchant implement to increase customer satisfaction for '{item_name}'?"
    else:
        user_question = f"Based on the best-selling item '{item_name}', what marketing or promotional strategy should be implemented?"

    # Ask Gemini to generate a response
    response = gemini_model.generate_content(user_question)

    # Return Gemini's response
    return response.text


# Chat loop
while True:
    user_question = input("Ask the assistant (or type 'exit' to quit): ")

    if user_question.lower() == "exit":
        break

    try:
        # Gemini generates response
        response = gemini_model.generate_content(user_question)
        generated_text = response.text
        print("\n🧠 Gemini's Answer:\n", generated_text)

        # Summary
        summary = summarizer(generated_text, max_length=50,
                             min_length=25, do_sample=False)
        print("\n📝 Summary:\n", summary[0]['summary_text'])

        # Sentiment classification
        classification_result = classifier(generated_text)
        print("\n📊 Sentiment:\n", classification_result)

    except Exception as e:
        print(f"\n⚠️ An error occurred: {e}")

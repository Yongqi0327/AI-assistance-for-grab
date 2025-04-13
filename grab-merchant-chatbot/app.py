import pandas as pd
import google.generativeai as genai
from transformers import pipeline
from deep_translator import GoogleTranslator


# 🌐 Gemini API Key
GOOGLE_API_KEY = "AIzaSyBv3OJ63w68FJYFhX1O3zsMOYSp7uWxwqs"
genai.configure(api_key=GOOGLE_API_KEY)

# 🧠 Load Gemini Model
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# 📊 Load Datasets
df_items = pd.read_csv("items.csv")
df_keywords = pd.read_csv("keywords.csv")
df_merchant = pd.read_csv("merchant.csv")
df_transaction_data = pd.read_csv("transaction_data.csv")
df_transaction_items = pd.read_csv("transaction_items.csv")

# Ensure column names are cleaned
for df in [df_items, df_keywords, df_merchant, df_transaction_data, df_transaction_items]:
    df.columns = df.columns.str.strip()

# 🛠️ ML Tools
summarizer = pipeline("summarization")
classifier = pipeline("text-classification", model="bert-base-uncased")

# 🔑 Keywords and Analysis Types
keywords = {
    'sales': ['sales', 'best-selling', 'top seller', 'quantity sold', 'revenue', 'income'],
    'promotion': ['promotion', 'discount', 'offer', 'coupon', 'deal'],
    'marketing': ['marketing', 'strategy', 'advertising', 'promotion', 'campaign'],
    'customer': ['customer', 'feedback', 'reviews', 'satisfaction', 'retention'],
    'trend': ['trend', 'pattern', 'growth', 'decline', 'seasonal'],
    'comparison': ['compare', 'versus', 'against', 'difference', 'performance'],
    'forecast': ['predict', 'forecast', 'future', 'projection', 'expect'],
    'inventory': ['inventory', 'stock', 'supply', 'shortage', 'overstocked'],
    'combo': ['combo', 'bundle', 'package', 'deal', 'set'],
    'price': ['price', 'cost', 'value', 'expensive', 'cheap']
}

# 🌍 Language Options
LANGUAGE_OPTIONS = {
    "1": ("English", "en"),
    "2": ("Malay", "ms"),
    "3": ("Indonesian", "id"),
    "4": ("Thai", "th"),
    "5": ("Vietnamese", "vi"),
    "6": ("Filipino", "tl")
}


def choose_language():
    print("🌐 Please choose your language:")
    for key, (name, _) in LANGUAGE_OPTIONS.items():
        print(f"{key}. {name}")
    choice = input("Enter number: ").strip()
    return LANGUAGE_OPTIONS.get(choice, ("English", "en"))

# 🔍 Keyword Matching


def match_keywords(user_question):
    matched_keywords = []
    for category, words in keywords.items():
        for word in words:
            if word.lower() in user_question.lower():
                matched_keywords.append(category)
                break
    # Default to sales if no keyword matched
    return matched_keywords if matched_keywords else ['sales']

# 🔍 Data Analysis Functions


def analyze_sales_data():
    """Analyze sales data and return key insights"""
    # Merge transaction items with transaction data
    merged_data = pd.merge(df_transaction_items,
                           df_transaction_data, on='order_id')

    # Join with items data to get item names
    sales_data = pd.merge(merged_data, df_items, on='item_id')

    # Calculate sales metrics
    item_sales = sales_data.groupby(['item_id', 'item_name']).agg(
        sales_count=('order_id', 'count'),
        revenue=('item_price', 'sum')
    ).reset_index().sort_values('sales_count', ascending=False)

    # Get top 5 items by sales
    top_items = item_sales.head(5)

    # Calculate total sales and revenue
    total_sales = item_sales['sales_count'].sum()
    total_revenue = item_sales['revenue'].sum()

    # Calculate sales by day of week (if date field exists)
    if 'order_date' in df_transaction_data.columns:
        df_transaction_data['day_of_week'] = pd.to_datetime(
            df_transaction_data['order_date']).dt.day_name()
        sales_by_day = df_transaction_data.groupby(
            'day_of_week').size().to_dict()
    else:
        sales_by_day = {"No date data available": 0}

    return {
        "top_items": top_items.to_dict('records'),
        "total_sales": total_sales,
        "total_revenue": total_revenue,
        "sales_by_day": sales_by_day,
        "item_count": len(df_items),
        "transaction_count": len(df_transaction_data)
    }


def analyze_customer_data():
    """Analyze customer data for insights"""
    # In a real app, we would analyze customer behavior
    # For this example, let's create some hypothetical metrics
    merged_data = pd.merge(df_transaction_items,
                           df_transaction_data, on='order_id')

    # Count unique customers
    if 'customer_id' in df_transaction_data.columns:
        unique_customers = df_transaction_data['customer_id'].nunique()
        orders_per_customer = df_transaction_data.groupby(
            'customer_id').size().mean()
    else:
        unique_customers = "Unknown"
        orders_per_customer = "Unknown"

    # Calculate average order value
    avg_order_value = merged_data.groupby(
        'order_id')['item_price'].sum().mean()

    return {
        "unique_customers": unique_customers,
        "orders_per_customer": orders_per_customer,
        "avg_order_value": avg_order_value
    }


def analyze_inventory_data():
    """Analyze inventory patterns"""
    # In a real system, we'd have inventory data
    # Let's create placeholder insights based on items and transactions

    # Calculate how frequently each item is ordered
    merged_data = pd.merge(df_transaction_items,
                           df_transaction_data, on='order_id')
    item_frequency = merged_data.groupby(
        'item_id').size().reset_index(name='order_count')
    item_frequency = pd.merge(
        item_frequency, df_items[['item_id', 'item_name']], on='item_id')

    # Identify items that might need inventory attention
    high_demand_items = item_frequency.nlargest(3, 'order_count')
    low_demand_items = item_frequency.nsmallest(3, 'order_count')

    return {
        "high_demand_items": high_demand_items.to_dict('records'),
        "low_demand_items": low_demand_items.to_dict('records'),
        "total_items": len(df_items)
    }


def analyze_trends():
    """Analyze trends over time"""
    # If we have date information, analyze trends
    if 'order_date' in df_transaction_data.columns:
        df_transaction_data['date'] = pd.to_datetime(
            df_transaction_data['order_date'])
        df_transaction_data['month'] = df_transaction_data['date'].dt.month
        df_transaction_data['day'] = df_transaction_data['date'].dt.day

        # Sales by month
        sales_by_month = df_transaction_data.groupby('month').size().to_dict()

        # Recent trend (last 30 days vs previous 30 days)
        # This would require more date manipulation in reality

        return {
            "sales_by_month": sales_by_month
        }
    else:
        return {"trend_data": "No date information available for trend analysis"}


def analyze_combo_potential():
    """Analyze potential for combo sales"""
    merged_data = pd.merge(pd.merge(df_transaction_items, df_transaction_data, on='order_id'),
                           df_items, on='item_id')

    # Find items that are commonly purchased together
    # Group by order_id to find items purchased in the same order
    order_groups = merged_data.groupby(
        'order_id')['item_id'].apply(list).reset_index()

    # Create a dictionary to store item pairs and their frequency
    item_pairs = {}
    top_combos = []

    # This is a simplified approach - in real production code, you'd use more efficient algorithms
    for _, row in order_groups.iterrows():
        items = row['item_id']
        if len(items) > 1:
            for i in range(len(items)):
                for j in range(i+1, len(items)):
                    pair = tuple(sorted([items[i], items[j]]))
                    if pair in item_pairs:
                        item_pairs[pair] += 1
                    else:
                        item_pairs[pair] = 1

    # Get top 3 most common pairs
    if item_pairs:
        sorted_pairs = sorted(item_pairs.items(),
                              key=lambda x: x[1], reverse=True)
        for pair, count in sorted_pairs[:3]:
            item1 = df_items[df_items['item_id']
                             == pair[0]]['item_name'].values[0]
            item2 = df_items[df_items['item_id']
                             == pair[1]]['item_name'].values[0]
            top_combos.append({
                "item1": item1,
                "item2": item2,
                "frequency": count
            })

    # Calculate average items per order
    avg_items_per_order = merged_data.groupby(
        'order_id')['item_id'].count().mean()

    return {
        "top_combos": top_combos,
        "avg_items_per_order": avg_items_per_order
    }


def get_best_selling_item():
    """Get the absolute best selling item details"""
    merged_data = pd.merge(df_transaction_items,
                           df_transaction_data, on='order_id')
    sales_data = pd.merge(merged_data, df_items, on='item_id')

    # Group by item and count occurrences
    item_sales = sales_data.groupby(['item_id', 'item_name']).agg(
        sales_count=('order_id', 'count'),
        revenue=('item_price', 'sum')
    ).reset_index().sort_values('sales_count', ascending=False)

    if len(item_sales) > 0:
        best_seller = item_sales.iloc[0]
        return {
            "name": best_seller['item_name'],
            "sales_count": int(best_seller['sales_count']),
            "revenue": float(best_seller['revenue']),
            "average_price": float(best_seller['revenue'] / best_seller['sales_count'])
        }
    else:
        return {
            "name": "Unknown",
            "sales_count": 0,
            "revenue": 0,
            "average_price": 0
        }

# 🌐 Translation


def translate_text(text, target_language='en'):
    try:
        # Attempt to translate the text
        translated = GoogleTranslator(
            source='auto', target=target_language).translate(text)
        return translated
    except Exception as e:
        # Handle the exception if an error occurs during translation
        print(f"⚠️ Error in translation: {e}")
        return text

# 🤖 Generate data-driven insights based on query intent


def generate_prompt_for_intent(query_intent, matched_categories=None):
    """Generate a specific prompt based on the query intent"""

    if query_intent == 'best_selling_item':
        best_seller = get_best_selling_item()
        prompt = f"""Based on the sales data analysis, provide ONLY the following information about the best-selling item:

ITEM DETAILS:
- Best-selling item: {best_seller['name']}
- Total units sold: {best_seller['sales_count']}
- Total revenue: ${best_seller['revenue']:.2f}
- Average price: ${best_seller['average_price']:.2f}

"""

    elif query_intent == 'combo_analysis':
        combo_data = analyze_combo_potential()
        sales_data = analyze_sales_data()

        prompt = f"""Based on the following transaction data analysis, provide specific advice about combo/bundle offerings:

COMBO/BUNDLE ANALYSIS:
- Average items per order: {combo_data['avg_items_per_order']:.2f}
- Top selling individual items:
  1. {sales_data['top_items'][0]['item_name']}: {sales_data['top_items'][0]['sales_count']} units
  2. {sales_data['top_items'][1]['item_name']}: {sales_data['top_items'][1]['sales_count']} units
  3. {sales_data['top_items'][2]['item_name']}: {sales_data['top_items'][2]['sales_count']} units

"""
        if combo_data['top_combos']:
            prompt += "Frequently purchased together:\n"
            for i, combo in enumerate(combo_data['top_combos'], 1):
                prompt += f"  {i}. {combo['item1']} + {combo['item2']}: {combo['frequency']} times\n"

        prompt += """
Based on this data, provide detailed recommendations for:
1. Specific combo/bundle deals this merchant should offer
2. Appropriate pricing strategies for these combos (discount percentages, etc.)
3. How to promote these bundles effectively
4. How to identify which items would create the most attractive and profitable bundles"""

    elif query_intent == 'sales_strategy':
        sales_data = analyze_sales_data()
        customer_data = analyze_customer_data()

        prompt = f"""Based on the following merchant data, provide comprehensive strategies to increase overall sales:

SALES OVERVIEW:
- Total transactions: {sales_data['transaction_count']}
- Total revenue: ${sales_data['total_revenue']:.2f}
- Top selling items:
  1. {sales_data['top_items'][0]['item_name']}: {sales_data['top_items'][0]['sales_count']} units, ${sales_data['top_items'][0]['revenue']:.2f} revenue
  2. {sales_data['top_items'][1]['item_name']}: {sales_data['top_items'][1]['sales_count']} units, ${sales_data['top_items'][1]['revenue']:.2f} revenue
  3. {sales_data['top_items'][2]['item_name']}: {sales_data['top_items'][2]['sales_count']} units, ${sales_data['top_items'][2]['revenue']:.2f} revenue

CUSTOMER INSIGHTS:
- Average order value: ${customer_data['avg_order_value']:.2f}

Provide specific, actionable strategies for:
1. Increasing average order value 
2. Maximizing sales of top-performing items
3. Effective promotions that could boost overall sales
4. Menu optimization suggestions
5. Marketing approaches specifically for a Grab merchant"""

    # Add default case for any other query intent including 'general_analysis', 'customer_insight', 'inventory_analysis'
    else:
        sales_data = analyze_sales_data()
        prompt = f"""Based on the merchant's sales data, provide a general analysis of their business:

GENERAL BUSINESS OVERVIEW:
- Total transactions: {sales_data['transaction_count']}
- Total revenue: ${sales_data['total_revenue']:.2f}
- Total items in inventory: {sales_data['item_count']}
- Top selling items:
  1. {sales_data['top_items'][0]['item_name']}: {sales_data['top_items'][0]['sales_count']} units
  2. {sales_data['top_items'][1]['item_name']}: {sales_data['top_items'][1]['sales_count']} units
  3. {sales_data['top_items'][2]['item_name']}: {sales_data['top_items'][2]['sales_count']} units

Based on this data, provide relevant business insights and recommendations.
"""

    return prompt

# 🧠 Understand user's query intent


def understand_query(user_question):
    """Determine what the user is specifically asking about"""
    user_question = user_question.lower()

    # More precise detection for best selling items
    if any(term in user_question for term in ['best selling', 'best-selling', 'top seller', 'best item', 'most popular', 'what is the best selling']):
        return 'best_selling_item'

    # More targeted keyword matching for other categories
    if any(term in user_question for term in ['combo', 'bundle', 'package', 'deal', 'items bought together']):
        return 'combo_analysis'

    if any(term in user_question for term in ['increase sales', 'boost sales', 'improve sales', 'sales strategy', 'how to sell more']):
        return 'sales_strategy'

    if any(term in user_question for term in ['customer', 'feedback', 'review', 'satisfaction']):
        return 'customer_insight'

    if any(term in user_question for term in ['inventory', 'stock', 'supply', 'out of stock']):
        return 'inventory_analysis'

    # Default to general sales analysis if no specific intent detected
    return 'general_analysis'


def ask_gemini_for_prediction(user_question, lang_code):
    # First understand what the user is asking about
    query_intent = understand_query(user_question)

    # Get matched keywords for additional context
    matched_categories = match_keywords(user_question)

    # Generate a specific prompt based on query intent
    prompt = generate_prompt_for_intent(query_intent, matched_categories)

    # Add a focused instruction based on query_intent
    if query_intent == 'best_selling_item':
        prompt += "\n\nIMPORTANT: Provide ONLY information about the best-selling item. Keep your answer concise and focused on just the data requested. Do not provide additional analysis or recommendations."

    # Ask Gemini for a response based on the generated prompt
    response = gemini_model.generate_content(prompt)
    response_text = response.text

    # Translate back to the user's language
    if lang_code != 'en':
        # First translate "Insights:" separately
        insights_translated = translate_text(
            "** Insights: **", target_language=lang_code)

        # Replace any occurrences of "** Insights: **" with the translated version
        response_text = response_text.replace(
            "** Insights: **", insights_translated)

        # Then translate the full response
        translated_response = translate_text(
            response_text, target_language=lang_code)
        return translated_response
    return response_text


# 🌏 Localized Labels & Sentiments
language_labels = {
    "en": {
        "summary": "Summary",
        "sentiment": "Sentiment",
        "prompt": "Ask a question or type 'exit' to quit",
        "positive": "Positive",
        "neutral": "Neutral",
        "negative": "Negative"
    },
    "ms": {
        "summary": "Rumusan",
        "sentiment": "Sentimen",
        "prompt": "Tanya soalan atau taip 'exit' untuk keluar",
        "positive": "Positif",
        "neutral": "Neutral",
        "negative": "Negatif"
    },
    "id": {
        "summary": "Ringkasan",
        "sentiment": "Sentimen",
        "prompt": "Ajukan pertanyaan atau ketik 'exit' untuk keluar",
        "positive": "Positif",
        "neutral": "Netral",
        "negative": "Negatif"
    },
    "th": {
        "summary": "สรุป",
        "sentiment": "ความรู้สึก",
        "prompt": "ถามคำถามหรือพิมพ์ 'exit' เพื่อออก",
        "positive": "เชิงบวก",
        "neutral": "เป็นกลาง",
        "negative": "เชิงลบ"
    },
    "vi": {
        "summary": "Tóm tắt",
        "sentiment": "Cảm xúc",
        "prompt": "Đặt câu hỏi hoặc nhập 'exit' để thoát",
        "positive": "Tích cực",
        "neutral": "Trung lập",
        "negative": "Tiêu cực"
    },
    "tl": {
        "summary": "Buod",
        "sentiment": "Sentimyento",
        "prompt": "Magtanong o i-type ang 'exit' upang lumabas",
        "positive": "Positibo",
        "neutral": "Neutral",
        "negative": "Negatibo"
    }
}

# 🗣️ Convert Hugging Face label to friendly text


def interpret_sentiment(label, lang_dict):
    if label == "LABEL_2":
        return lang_dict["positive"]
    elif label == "LABEL_1":
        return lang_dict["neutral"]
    elif label == "LABEL_0":
        return lang_dict["negative"]
    else:
        return label


# 💬 Start Chat
language_name, lang_code = choose_language()
labels = language_labels.get(
    lang_code, language_labels["en"])  # fallback to English
print(f"\n✅ You selected: {language_name} ({lang_code})")
print(f"{labels['prompt']}")

while True:
    # Always display prompt before accepting user input
    user_input = input(f"\n{language_name} ➤ ")
    if user_input.lower() == "exit":
        print("Thanks for using Grabi. Have a nice day!")
        break

    # Translate user's input to English for Gemini if not already in English
    if lang_code != 'en':
        translated_input = translate_text(user_input, target_language="en")
    else:
        translated_input = user_input

    try:
        # Ask Gemini for insights
        response = ask_gemini_for_prediction(translated_input, lang_code)

        # The response is already translated in ask_gemini_for_prediction
        print(f"\n🧠 Grabi's Answer ({language_name}):\n{response}\n")

        # Truncate response to avoid model input limit
        response_short = response[:1024]

        # 📝 Summarization - MODIFY to suppress warnings
        try:
            # Adjust max_length based on input length to avoid warning
            input_length = len(response_short.split())
            # Dynamic max_length based on input
            max_length = min(60, max(25, input_length // 2))

            summary = summarizer(
                response_short, max_length=max_length, min_length=25, do_sample=False)
            summary_text = summary[0]['summary_text']

            # Translate summary if not in English
            if lang_code != 'en':
                translated_summary = translate_text(
                    summary_text, target_language=lang_code)
                print(f"\n📝 {labels['summary']}:\n{translated_summary}")
            else:
                print(f"\n📝 {labels['summary']}:\n{summary_text}")

        except Exception as e:
            # Don't print summarization errors to the user
            pass

        # 📊 Sentiment Analysis
        try:
            sentiment = classifier(response_short)
            # Use interpret_sentiment with localized labels
            sentiment_text = interpret_sentiment(sentiment[0]['label'], labels)
            print(f"\n📊 {labels['sentiment']}:\n{sentiment_text}")
        except Exception as e:
            # Don't print sentiment analysis errors to the user
            pass

    except Exception as e:
        # For API and other critical errors, we still want to show these
        print(f"⚠️ Error in processing: {e}")

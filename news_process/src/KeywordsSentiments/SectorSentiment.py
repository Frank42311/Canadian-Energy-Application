import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# Function to initialize the sentiment analysis model and tokenizer
# This function loads a pre-trained model and tokenizer designed for financial sentiment analysis.
def init_sentiment_model() -> (AutoTokenizer, AutoModelForSequenceClassification, torch.device):
    """
    Initializes and returns the sentiment analysis model, tokenizer, and device.

    Returns:
        tokenizer (AutoTokenizer): The tokenizer for the model.
        model (AutoModelForSequenceClassification): The sentiment analysis model.
        device (torch.device): The device (CUDA or CPU) the model is using.
    """
    print("Initializing sentiment analysis model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('ahmedrachid/FinancialBERT-Sentiment-Analysis')
    model = AutoModelForSequenceClassification.from_pretrained('ahmedrachid/FinancialBERT-Sentiment-Analysis')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    return tokenizer, model, device


# Function to analyze the sentiment of a given text
# It takes a string of text and the previously initialized model, tokenizer, and device to predict the sentiment score.
def analyze_sentiment(text: str, tokenizer: AutoTokenizer, model: AutoModelForSequenceClassification,
                      device: torch.device) -> float:
    """
    Analyzes the sentiment of a given text and returns a sentiment score.

    Parameters:
        text (str): The text to analyze.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        model (AutoModelForSequenceClassification): The sentiment analysis model.
        device (torch.device): The device the model is using.

    Returns:
        final_score (float): The sentiment score of the text.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
        final_score = prediction[0][1].item() - prediction[0][0].item()
    return round(final_score, 3)


tokenizer, model, device = init_sentiment_model()

# Paths for input and output directories
input_dir = "../../data/News/KeywordsSentiments/SectorKeywords"
output_dir = "../../data/News/KeywordsSentiments/SectorSentiments"
os.makedirs(output_dir, exist_ok=True)

# Processing files from 2008 to 2023
for year in range(2008, 2024):
    file_name = f"{year}_keywords.csv"
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, file_name)
    if os.path.exists(input_path):
        print(f"Processing file: {input_path}")
        df = pd.read_csv(input_path, encoding='utf-8')
        if not df.empty and 'keywords' in df.columns and 'sector_name' in df.columns:
            # Analyze sentiment of the 'keywords' column
            df['keywords_sentiment'] = df['keywords'].apply(
                lambda x: analyze_sentiment(x, tokenizer, model, device) if pd.notnull(x) else None)
            # Calculate the mean sentiment score for each sector
            sector_sentiment = df.groupby('sector_name')['keywords_sentiment'].mean()
            df['sector_sentiment'] = sector_sentiment
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"Saved sentiment scores to: {output_path}")
        else:
            print(f"Required columns not found in: {input_path}")
    else:
        print(f"File does not exist: {input_path}")

print("Sentiment analysis completed and saved.")

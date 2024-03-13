import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def init_sentiment_model() -> (AutoTokenizer, AutoModelForSequenceClassification, torch.device):
    """
    Initializes the sentiment analysis model and tokenizer using a pretrained model.
    It sets the device to GPU if available, otherwise to CPU.

    Returns:
        tokenizer (AutoTokenizer): The tokenizer for the model.
        model (AutoModelForSequenceClassification): The pretrained sentiment analysis model.
        device (torch.device): The device (GPU/CPU) the model will run on.
    """
    print("Initializing sentiment analysis model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('ahmedrachid/FinancialBERT-Sentiment-Analysis')
    model = AutoModelForSequenceClassification.from_pretrained('ahmedrachid/FinancialBERT-Sentiment-Analysis')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Using device:", device)
    return tokenizer, model, device

def analyze_sentiment(text: str, tokenizer: AutoTokenizer, model: AutoModelForSequenceClassification, device: torch.device) -> float:
    """
    Analyzes the sentiment of a given text string using the specified tokenizer, model, and device.

    Parameters:
        text (str): The text to analyze.
        tokenizer (AutoTokenizer): The tokenizer for preprocessing text.
        model (AutoModelForSequenceClassification): The sentiment analysis model.
        device (torch.device): The device (GPU/CPU) to perform the analysis on.

    Returns:
        final_score (float): The sentiment score of the text, ranging from -1 (negative) to 1 (positive).
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
        final_score = prediction[0][1].item() - prediction[0][0].item()
    return round(final_score, 3)

# Initialize sentiment analysis model, tokenizer, and device.
tokenizer, model, device = init_sentiment_model()

input_dir = "../../data/News/KeywordsSentiments/CompanyKeywords"
output_dir = "../../data/News/KeywordsSentiments/CompanySentiments"
os.makedirs(output_dir, exist_ok=True)

# Process files from 2008 to 2023
for year in range(2008, 2023 + 1):
    file_name = f"{year}_keywords.csv"
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, file_name)
    if os.path.exists(input_path):
        print(f"Processing file: {input_path}")
        df = pd.read_csv(input_path, encoding='utf-8')
        if not df.empty and 'keywords' in df.columns and 'company_name' in df.columns:
            # Analyze sentiment of the keywords and calculate average sentiment per company
            df['keywords_sentiment'] = df['keywords'].apply(
                lambda x: analyze_sentiment(x, tokenizer, model, device) if pd.notnull(x) else None)
            company_sentiment = df.groupby('company_name')['keywords_sentiment'].mean()
            df['company_sentiment'] = company_sentiment
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"Saved sentiment scores to: {output_path}")
        else:
            print("Required columns not found in the file.")
    else:
        print(f"File does not exist: {input_path}")

print("Sentiment analysis completed and results saved.")

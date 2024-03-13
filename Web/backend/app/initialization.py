from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from openai import OpenAI
from transformers import BertTokenizer, BertModel
import nltk

# Download necessary datasets for nltk
nltk.download('punkt')
nltk.download('stopwords')

# Setup OpenAI API
api_key_file_path = './gpt_api/api_key.txt'
with open(api_key_file_path, 'r') as file:
    api_key = file.read().strip()
client = OpenAI(api_key=api_key)

# Load pretrained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


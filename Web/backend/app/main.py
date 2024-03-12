from openai import OpenAI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Load pretrained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Download necessary datasets for nltk
nltk.download('punkt')
nltk.download('stopwords')

app = FastAPI()

# Serve static files from the 'frontend' directory
app.mount("/static", StaticFiles(directory="app/frontend"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    Serves the main HTML file to the client. This endpoint is the entry point of the web application,
    where the frontend interface is loaded.

    Returns:
    HTMLResponse: The HTML content of the main page loaded from a file, along with a 200 OK status code.
    """
    with open(Path('app/frontend/frontend.html'), 'r', encoding='utf-8') as html_file:
        return HTMLResponse(content=html_file.read(), status_code=200)

"""
Initializes the OpenAI API client with an API key.

The API key is read from a text file located at '../../gpt_api/api_key.txt'.
This key is used to authenticate requests to the OpenAI API, enabling access to models like GPT-4.
"""
# Setup OpenAI API
api_key_file_path = 'app/gpt_api/api_key.txt'
with open(api_key_file_path, 'r') as file:
    api_key = file.read().strip()
client = OpenAI(api_key=api_key)

def get_sentence_embedding(sentence: str) -> torch.Tensor:
    """
    Generates a sentence embedding using the BERT model.

    Parameters:
    - sentence (str): The sentence to embed.

    Returns:
    torch.Tensor: The embedding of the input sentence as a PyTorch tensor.
    """
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    sentence_embedding = outputs.last_hidden_state[:, 0, :]
    return sentence_embedding

def calculate_sentence_similarity(sentences: List[str]) -> np.ndarray:
    """
    Calculates the cosine similarity between multiple sentences by first generating embeddings for each sentence
    using the BERT model and then calculating the pairwise cosine similarity.

    Parameters:
    - sentences (List[str]): A list of sentences to compare.

    Returns:
    np.ndarray: A matrix of cosine similarity scores between the sentences.
    """
    embeddings = [get_sentence_embedding(sentence) for sentence in sentences]
    similarity_matrix = cosine_similarity(torch.stack(embeddings).squeeze().numpy())
    return similarity_matrix

def merge_closest_paragraphs(paragraphs: List[str], target_n_paragraphs: int) -> List[str]:
    """
    Merges paragraphs to reduce the total count to the target number, prioritizing the merge of paragraphs
    with the highest similarity to each other.

    Parameters:
    - paragraphs (List[str]): The initial list of paragraphs.
    - target_n_paragraphs (int): The target number of paragraphs after merging.

    Returns:
    List[str]: The modified list of paragraphs, with the total count reduced to the target number.
    """
    while len(paragraphs) > target_n_paragraphs:
        # Calculate paragraph similarities
        paragraph_embeddings = [get_sentence_embedding(paragraph) for paragraph in paragraphs]
        similarity_matrix = cosine_similarity(torch.stack(paragraph_embeddings).squeeze().numpy())

        # Find the pair of paragraphs with the highest similarity
        max_similarity, merge_index = 0, 0
        for i in range(len(similarity_matrix) - 1):
            if similarity_matrix[i, i+1] > max_similarity:
                max_similarity = similarity_matrix[i, i+1]
                merge_index = i

        # Merge the pair of paragraphs with the highest similarity
        paragraphs[merge_index] = paragraphs[merge_index] + " " + paragraphs[merge_index + 1]
        del paragraphs[merge_index + 1]

    return paragraphs

def split_query_into_paragraphs(text: str, images_count: int, similarity_threshold: float = 0.9) -> List[str]:
    """
    Splits the input text into paragraphs based on sentence similarity, with the goal of creating a number of paragraphs
    that matches the specified target, influenced by the number of images.

    Parameters:
    - text (str): The input text to split into paragraphs.
    - images_count (int): The number of images, which influences the target number of paragraphs.
    - similarity_threshold (float): The threshold for considering sentences similar enough to be in the same paragraph.

    Returns:
    List[str]: A list of paragraphs constructed from the input text.
    """
    sentences = sent_tokenize(text)
    n_paragraphs = images_count + 1

    if len(sentences) <= n_paragraphs:
        return [' '.join(sentences)] * n_paragraphs

    # Initial paragraph segmentation
    paragraphs = []
    current_paragraph = sentences[0]
    for i in range(1, len(sentences)):
        if calculate_sentence_similarity([current_paragraph, sentences[i]])[0, 1] >= similarity_threshold:
            current_paragraph += " " + sentences[i]
        else:
            paragraphs.append(current_paragraph)
            current_paragraph = sentences[i]
    paragraphs.append(current_paragraph)

    # Merge closest paragraphs if there are more paragraphs than the target
    if len(paragraphs) > n_paragraphs:
        paragraphs = merge_closest_paragraphs(paragraphs, n_paragraphs)

    return paragraphs

class QueryItem(BaseModel):
    """
    A Pydantic model that defines the structure for query items submitted to the API.

    Attributes:
    - query (str): The main text query from the user.
    - sector (str): The sector or category of the query (unused in the given code but may be intended for future use).
    - source (str): The source of data ('Online' for querying OpenAI's model directly or another keyword for local processing).
    - images (Optional[List[str]]): An optional list of base64-encoded images associated with the query.
    """
    query: str
    sector: str
    source: str
    images: Optional[List[str]] = None



def get_gpt_response(prompt):
    """
    Sends a prompt to the OpenAI API and returns the GPT-4 model's response.

    Parameters:
    - prompt (str): The input text to send to the model.

    Returns:
    - str: The model's response as a string. If an exception occurs, returns the error message.
    """
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="gpt-4", # Use gpt-4 model
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return str(e)

def analyze_image_with_openai(base64_image, api_key):
    """
    Analyzes an image using OpenAI's GPT-4 Vision model by sending a base64-encoded image.

    Parameters:
    - base64_image (str): The base64-encoded image to analyze.
    - api_key (str): The API key for authentication with OpenAI.

    Returns:
    - str: The analysis of the image. If an error occurs, returns the error message.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [{
                "role": "user",
                "content":
                    [{
                        "type": "text",
                        "text": "What’s in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"}
                }]
        }],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    # Parse the response to get the 'content' part
    try:
        response.raise_for_status()  # Check for HTTP errors first

        # Extract the 'content' field from the first 'message' in 'choices'
        content = response.json()['choices'][0]['message']['content']

    except KeyError:
        content = "Error parsing response: Key not found."
    except requests.exceptions.HTTPError as http_err:
        content = f"HTTP error occurred: {http_err}"
    except Exception as err:
        content = f"An error occurred: {err}"

    return content


def load_data(pickle_file_path):
    """
    Loads and returns data from a pickle file.

    Parameters:
    - pickle_file_path (str): The path to the pickle file to load.

    Returns:
    - The data loaded from the pickle file.
    """
    with open(pickle_file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def preprocess(text):
    """
    Preprocesses a given text by tokenizing, removing stopwords, and filtering out non-alphanumeric words.

    Parameters:
    - text (str): The text to preprocess.

    Returns:
    - str: The preprocessed text.
    """
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_text = [word for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_text)


def find_top_similar_paragraphs(preprocessed_paragraph, vectorizer, tfidf_matrix, documents, top_n=10):
    """
    Finds and returns the top N most similar paragraphs to a given preprocessed paragraph.

    Parameters:
    - preprocessed_paragraph (str): The preprocessed paragraph to compare against the document collection.
    - vectorizer: The vectorizer used to convert text into vector space.
    - tfidf_matrix: The TF-IDF matrix representing the document collection.
    - documents (list): The list of original documents.
    - top_n (int): The number of top similar paragraphs to return.

    Returns:
    - str: Concatenated string of the top N most similar paragraphs.
    """
    # Vectorize the query paragraph
    query_vector = vectorizer.transform([preprocessed_paragraph])

    # Compute similarity
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Get the indices of the top 10 most similar documents
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]

    # Build the return result
    similar_paragraphs = [f"Article {i + 1}\n{documents[idx]}\n" for i, idx in enumerate(top_indices)]
    return "".join(similar_paragraphs)

def generate_prompt(final_paragraph, similar_paragraphs):
    """
   Generates a prompt for the GPT model based on a final paragraph and a list of similar paragraphs.

   Parameters:
   - final_paragraph (str): The final paragraph to include in the prompt.
   - similar_paragraphs (str): Concatenated string of similar paragraphs to include in the prompt.

   Returns:
   - str: The generated prompt.
   """
    prompt = f"Based on the document descriptions provided, please answer the following question related to the content: \n\n'{final_paragraph}'\n\n"
    prompt += "Relevant document excerpts:\n"
    prompt += similar_paragraphs
    return prompt


async def analyze_images(images: Optional[List[str]], api_key: str) -> List[str]:
    """
    Analyzes a list of images using the OpenAI API and returns their descriptions.

    Parameters:
    - images (Optional[List[str]]): A list of base64-encoded images to analyze.
    - api_key (str): The API key for authenticating with the OpenAI API.

    Returns:
    - List[str]: A list of descriptions for each image.
    """
    image_responses = []
    for base64_image in images:
        # images from frontend are in base64 format
        image_analysis_response = analyze_image_with_openai(base64_image.split(",")[-1], api_key)
        image_responses.append(image_analysis_response)
    return image_responses

async def process_text_query(query: str, source: str, api_key: str, data_path: str) -> str:
    """
    Processes a text query based on the source specified (Online or local data).

    Parameters:
    - query (str): The text query to process.
    - source (str): The source of the data ('Online' for GPT-3 or local dataset).
    - api_key (str): The API key for authenticating with the OpenAI API.
    - data_path (str): The path to the local dataset (if source is not 'Online').

    Returns:
    - str: The response to the query.
    """
    if source == 'Online':
        return get_gpt_response(query)
    else:
        data = load_data(data_path)
        preprocessed_query = preprocess(query)
        similar_paragraphs = find_top_similar_paragraphs(preprocessed_query, data['vectorizer'], data['tfidf_matrix'], data['documents'])
        prompt = generate_prompt(query, similar_paragraphs)
        return get_gpt_response(prompt)

@app.post("/ask")
async def ask_gpt(item: QueryItem):
    """
    Endpoint to handle queries and generate responses based on text and/or images.

    This function analyzes images and processes text queries to generate a comprehensive response
    combining insights from both text and image analyses.

    Parameters:
    - item (QueryItem): An object containing the query text, sector, source, and optional images.

    Returns:
    - JSON: A JSON object containing the generated response or an error message.
    """
    responses = []

    # Handle image analysis
    images_count = 0 if item.images is None else len(item.images)
    if images_count > 0:
        image_responses = await analyze_images(item.images, api_key)
        responses.extend(image_responses)

    # Process text query
    text_response = await process_text_query(item.query, item.source, api_key, 'data/tfidf_vectorizer.pkl')
    responses.append(text_response)

    # Generate final response
    final_gpt_response = " ".join(responses) if images_count > 0 else text_response
    if final_gpt_response:
        return {"response": final_gpt_response}
    else:
        raise HTTPException(status_code=400, detail="Failed to get final response")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
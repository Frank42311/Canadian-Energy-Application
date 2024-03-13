import nltk
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from initialization import tokenizer, model
from typing import List

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
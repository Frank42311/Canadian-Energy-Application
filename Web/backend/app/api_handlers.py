from fastapi import APIRouter, HTTPException

from initialization import api_key
from openai_utils import analyze_image_with_openai, get_gpt_response
from models import QueryItem
from typing import Optional, List

from text_processing import load_data, preprocess, find_top_similar_paragraphs, generate_prompt

router = APIRouter()

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

@router.post("/ask")
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

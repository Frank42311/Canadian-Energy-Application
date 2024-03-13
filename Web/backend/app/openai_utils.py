import requests

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
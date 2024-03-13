import json
from typing import Tuple, Dict, Union


def extract_root_domain(url: str) -> str:
    """
    Extracts the root domain from a URL.

    Parameters:
    - url (str): The URL from which the root domain needs to be extracted.

    Returns:
    - str: The extracted root domain.
    """
    # Determine the starting point of the domain name
    if 'www.' in url:
        start = url.find('www.') + 4
    else:
        start = url.find('//') + 2

    # Find the end of the domain name, before the path starts
    end = url.find('/', start)

    # Extract and return the domain name
    return url[start:end] if end != -1 else url[start:]


def has_three_consecutive_chars(str1: str, str2: str) -> bool:
    """
    Checks if there are at least three consecutive matching characters in two strings.

    Parameters:
    - str1 (str): The first string to compare.
    - str2 (str): The second string to compare.

    Returns:
    - bool: True if there are at least three consecutive matching characters, False otherwise.
    """
    # Loop through str1 to find any sequence of 3 characters that occurs in str2
    for i in range(len(str1) - 2):
        if str1[i:i + 3] in str2:
            return True
    return False


def process_json_file(input_path: str, output_path: str) -> None:
    """
    Processes a JSON file to extract root domains from URLs and check for matching character sequences.

    Parameters:
    - input_path (str): Path to the input JSON file containing URLs.
    - output_path (str): Path where the modified JSON file will be saved.

    Returns:
    - None
    """
    # Open and read the input JSON file
    with open(input_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    modified_data: Dict[str, Union[str, list]] = {}

    # Process each key-value pair in the JSON data
    for key, url in data.items():
        root_domain = extract_root_domain(url)
        # Print key and URL if they don't share a sequence of 3 consecutive characters
        if not has_three_consecutive_chars(key, root_domain):
            print(f"Key: {key}, URL: {url}")

        # Add the processed data to a new dictionary
        modified_data[key] = [url, root_domain]

    # Write the modified data to the output JSON file
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(modified_data, file, indent=4)


# Define the paths to your input and output JSON files
input_path = '../../data/CompanyInformation/JSON/company_official_site.json'
output_path = '../../data/CompanyInformation/JSON/company_official_site_revised.json'

# Call the function to process the file
process_json_file(input_path, output_path)

import json
import pandas as pd
import os
from openai import OpenAI
import time


def load_json_data(json_file: str) -> list:
    """
    Load a JSON file and return a list of tuples, each containing a sector name and a set of company names.

    Parameters:
    - json_file (str): The path to the JSON file.

    Returns:
    - list: A list of tuples, where each tuple contains a sector name (str) and a set of company names (set).
    """
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return [(sector, set(companies)) for sector, companies in data.items()]


def add_sector_and_filter_summary(year: int, json_data: list, api_key: str) -> None:
    """
    Read a CSV file for the specified year, add a 'sector_name' column, filter the data based on summary content,
    and save to a new CSV file.

    Parameters:
    - year (int): The year of the CSV file to process.
    - json_data (list): The sector and company information loaded from a JSON file.
    - api_key (str): The API key for accessing OpenAI's API.
    """
    input_path = os.path.join('../../data/News/Summarized/Articles', f'{year}.csv')
    output_path = os.path.join('../../data/News/Summarized/ArticlesCleaned', f'{year}.csv')
    error_log_path = '../../data/News/Summarized/ArticlesCleaned/Error.txt'
    invalid_msgs_path = '../../data/News/Summarized/ArticlesCleaned/InvalidMsgs.txt'

    df = pd.read_csv(input_path, encoding='utf-8')
    df['sector_name'] = ''  # Initialize the 'sector_name' column.

    # Adding 'sector_name' column.
    for sector, companies in json_data:
        df['sector_name'] = df.apply(lambda x: sector if x['company_name'] in companies else x['sector_name'], axis=1)

    # Initialize OpenAI client.
    client = OpenAI(api_key=api_key)

    # Generate prompt and call the API.
    for index, row in df.iterrows():

        # Check if summary is NaN.
        if pd.isna(row['summary']):
            continue

        prompt = f"Is this article related to financial news or related to {row['sector_name']} industry? Answer 'yes' or 'no': {row['summary']}"
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="gpt-3.5-turbo-1106",
            )
            response = chat_completion.choices[0].message.content

            # Process API response.
            if 'no' in response:
                df.at[index, 'summary'] = ''
                df.at[index, 'success'] = -2
                # Log invalid messages.
                with open(invalid_msgs_path, 'a', encoding='utf-8') as file:
                    file.write(f"{year}\t{index}\t{row['summary']}\n")

        except Exception as e:
            # Log error information.
            with open(error_log_path, 'a', encoding='utf-8') as file:
                file.write(f"Error at year {year}, index {index}: {str(e)}\n")
            time.sleep(60)  # Pause for 60 seconds.
            try:
                # Retry API call.
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model="gpt-3.5-turbo-1106",
                )
                response = chat_completion.choices[0].message.content
                if 'no' in response:
                    df.at[index, 'summary'] = ''
                    df.at[index, 'success'] = -2
                    with open(invalid_msgs_path, 'a', encoding='utf-8') as file:
                        file.write(f"{year}\t{index}\t{row['summary']}\n")
            except Exception as e:
                # If retry fails, log the error and continue to the next row.
                with open(error_log_path, 'a', encoding='utf-8') as file:
                    file.write(f"Same error second time at year {year}, index {index}: {str(e)}\n")
                continue

    df.to_csv(output_path, index=False, encoding='utf-8')


json_file = '../../data/CompanyInformation/JSON/company_info.json'
json_data = load_json_data(json_file)
api_key_file_path = '../../gpt_api/api_key.txt'

# Read the API token.
with open(api_key_file_path, 'r') as file:
    api_key = file.read().strip()

# Assuming you need to process CSV files for multiple years.
for year in range(2008, 2024):
    print(f'Processing: {year}')
    add_sector_and_filter_summary(year, json_data, api_key)

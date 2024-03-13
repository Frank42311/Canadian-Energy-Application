import pandas as pd
import os
from openai import OpenAI

# Set up the OpenAI API by reading the API key from a file
api_key_file_path = '../../gpt_api/api_key.txt'
with open(api_key_file_path, 'r') as file:
    api_key = file.read().strip()  # Read and trim any leading/trailing whitespace
client = OpenAI(api_key=api_key)


def get_gpt_response(prompt: str) -> str:
    """
    Call the GPT model and handle any errors that occur.

    Parameters:
    - prompt (str): The prompt to send to the GPT model.

    Returns:
    - str: The response from GPT, or an error message if an error occurs.
    """
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="gpt-3.5-turbo-1106",
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return str(e)


def segment_content(content: str, max_length: int = 3000) -> list:
    """
    Segment content into smaller parts if it exceeds a maximum length.

    Parameters:
    - content (str): The content to segment.
    - max_length (int): The maximum length of each segment.

    Returns:
    - list: A list of segmented content.
    """
    segments = []
    current_segment = ''
    for part in content.split('###'):
        if len(current_segment) + len(part) + 3 > max_length:  # +3 for "###"
            segments.append(current_segment)
            current_segment = part
        else:
            if current_segment:
                current_segment += '###'
            current_segment += part
    if current_segment:
        segments.append(current_segment)
    return segments


def process_data(file_path: str) -> pd.DataFrame:
    """
    Process data from a CSV file, combining and summarizing the summaries for each company.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - pd.DataFrame: The final DataFrame with combined and summarized data.
    """
    # Read the CSV file, ensuring proper text encoding
    df = pd.read_csv(file_path, encoding='utf-8')
    df = df[['company_name', 'summary', 'sector_name']]
    df['summary'].replace('', pd.NA, inplace=True)

    # Combine summaries for each company by sector
    df_combined = df.groupby(['company_name', 'sector_name'])['summary'].apply(
        lambda x: '###'.join(x.dropna())).reset_index()

    # Process each combined summary
    for index, row in df_combined.iterrows():
        content = row['summary']
        if len(content) > 3000:
            # If content exceeds 3000 characters, segment and summarize each part
            segments = segment_content(content)
            prompts = [f"Summarize this detailed content: {segment}" for segment in segments]
            summaries = [get_gpt_response(prompt) for prompt in prompts]
            consolidated_summary = ' '.join(summaries)
            final_summary = get_gpt_response(f"Provide a final summary: {consolidated_summary}")
        else:
            # Directly summarize content if under the length limit
            prompt = f"Summarize this detailed content: {content}"
            final_summary = get_gpt_response(prompt)

        # Update DataFrame with the final summaries
        df_combined.loc[index, 'summary_result'] = final_summary

    # Rearrange DataFrame to include the new 'summary_result' column
    final_df = df_combined[['company_name', 'summary', 'sector_name', 'summary_result']]
    return final_df

# Example usage for years 2008 to 2023
for year in range(2008, 2024):
    file_path = f'../../data/News/Summarized/ArticlesCleaned/{year}.csv'  # Corrected file path for compatibility
    final_df = process_data(file_path)
    # Save the processed data to a CSV file
    final_df.to_csv(f'../../data/News/Summarized/Companies/{year}.csv', index=False, encoding='utf-8')

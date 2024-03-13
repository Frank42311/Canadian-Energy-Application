import pandas as pd
import os
from openai import OpenAI

# Set up the OpenAI API by reading the API key from a file.
api_key_file_path = '../../gpt_api/api_key.txt'
with open(api_key_file_path, 'r') as file:
    api_key = file.read().strip()
client = OpenAI(api_key=api_key)


def get_gpt_response(prompt: str) -> str:
    """
    Calls the GPT model and returns its response.

    Parameters:
    - prompt (str): The text prompt to send to the GPT model.

    Returns:
    - str: The GPT model's response or an error message.
    """
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-3.5-turbo-1106",
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return str(e)


def segment_content(sector_name: str, content: str) -> (list, list):
    """
    Splits the content into manageable segments and generates prompts for GPT based on those segments.

    Parameters:
    - sector_name (str): The name of the sector for which the content is being segmented.
    - content (str): The entire content to be segmented.

    Returns:
    - A tuple of two lists:
        1. The segments of content.
        2. The corresponding prompts for each segment.
    """
    words = content.split()
    segments = [content] if len(words) <= 3000 else []
    current_segment, current_count = [], 0

    if len(words) > 3000:
        for word in words:
            if word == "###" and current_count >= 3000:
                segments.append(" ".join(current_segment))
                current_segment, current_count = [], 0
            current_segment.append(word)
            current_count += 1
        if current_segment:  # Ensure the last segment is added
            segments.append(" ".join(current_segment))

    prompts = [
        f"Provide a comprehensive summary focusing on the {sector_name} sector for the year, highlighting key developments, challenges, and achievements. Pay special attention to advancements in technology, market dynamics, regulatory changes, and significant industry milestones. Ensure the report captures the essence of the sector's progress while discussing its difficulties, including economic fluctuations, supply chain issues, and environmental challenges."
        for _ in segments]

    return segments, prompts


def consolidate_summaries(sector_name: str, year: int, summaries: list) -> str:
    """
    Consolidates individual summaries into a final, comprehensive summary.

    Parameters:
    - sector_name (str): The sector name for the summaries.
    - year (int): The year of the summaries.
    - summaries (list): A list of strings, each a summary to be consolidated.

    Returns:
    - str: A consolidated summary.
    """
    consolidated_content = " ".join(
        [f"For the {sector_name} sector, summary part {i + 1}: {summary}" for i, summary in enumerate(summaries)])
    final_prompt = f"Provide a concise summary for the {sector_name} sector for {year}, not exceeding 150 words, based on the following summaries."
    final_summary = get_gpt_response(final_prompt + consolidated_content)
    return final_summary


# Set up source and target directories for data processing.
source_folder = '../../data/News/Summarized/Companies'
target_folder = '../../data/News/Summarized/Sectors'
os.makedirs(target_folder, exist_ok=True)

# Process each year's data file.
for year in range(2008, 2023):
    file_name = f"{year}.csv"
    source_file_path = os.path.join(source_folder, file_name)
    target_file_path = os.path.join(target_folder, file_name)

    try:
        df = pd.read_csv(source_file_path, encoding='utf-8')
        rows = []

        for sector_name, group in df.groupby('sector_name'):
            sector_content = " ".join(group['summary_result'].tolist())
            segments, prompts = segment_content(sector_name, sector_content)

            summaries = [get_gpt_response(prompt) for prompt in prompts]
            final_summary = consolidate_summaries(sector_name, year, summaries)

            rows.append({'sector_name': sector_name, 'sector_summary': final_summary})

        consolidated_df = pd.DataFrame(rows)
        consolidated_df.to_csv(target_file_path, index=False)
        print(f"Processed and saved: {file_name}")
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

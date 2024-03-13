import os
import pandas as pd
import numpy as np


def filter_length(content: str) -> str:
    """
    Filters out content whose length does not meet specified conditions.

    Parameters:
    - content: str, the content to be filtered.

    Returns:
    - The original content if its length is between 200 and 30000 characters, inclusive; otherwise, np.nan.
    """
    length = len(str(content))
    return content if 200 <= length <= 30000 else np.nan


    """
    Removes specific strings from the content.

    If the content contains any of the targeted strings (e.g., "cookie", "login"),
    this function will remove the entire line containing that string.

    Parameters:
    - content: str, the content from which specific strings are to be removed.

    Returns:
    - The content with specified strings removed. If the content is NaN, it returns NaN.
    """
    if pd.isna(content):
        return content
    for string in ["cookie", " login ", " log in ", "register for free"]:
        start_index = content.lower().find(string)
        while start_index != -1:
            end_index = content.find("\n", start_index)
            if end_index != -1:
                content = content[:start_index] + content[end_index + 1:]
            else:
                content = content[:start_index]
            start_index = content.lower().find(string)
    return content


def clean_content(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the data by applying a series of filters:
    - Filters by content length
    - Removes specific unwanted strings
    - Filters by content length again

    Parameters:
    - data: pd.DataFrame, a dataframe with a column 'content_or_error' that contains the text to be cleaned.

    Returns:
    - The dataframe after the cleaning process, with the 'success' column updated based on whether the content was cleaned successfully.
    """
    # First step: filter by character count
    data['content_or_error'] = data['content_or_error'].apply(filter_length)
    data.loc[data['content_or_error'].isna(), 'success'] = -1

    # Second step: remove specific strings
    data['content_or_error'] = data['content_or_error'].apply(remove_specific_strings)

    # Third step: filter by character count again
    data['content_or_error'] = data['content_or_error'].apply(filter_length)
    data.loc[data['content_or_error'].isna(), 'success'] = -1

    return data


def main(year: int) -> None:
    """
    Main function to clean content of news articles for a given year.

    - Reads a CSV file for the specified year.
    - Cleans the content of the articles.
    - Saves the cleaned content to a new CSV file.

    Parameters:
    - year: int, the year for which the news content will be cleaned.
    """
    # Construct the original file path
    file_path = os.path.join('..', '..', 'data', 'News', 'Initial', 'Cleaned', f'{year}.csv')
    # Construct the save file path
    save_path = os.path.join('..', '..', 'data', 'News', 'Initial', 'FurtherCleaned', f'{year}.csv')

    try:
        # Read the CSV file
        data = pd.read_csv(file_path, encoding='utf-8')

        # Clean the data
        data = clean_content(data)

        # Save to a new CSV file
        data.to_csv(save_path, index=False, encoding='utf-8')

    except Exception as e:
        print(f"Error: {e}")


# Example usage with a range of years
for year in range(2000, 2024):
    main(year)

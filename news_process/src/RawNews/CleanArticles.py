import pandas as pd
import os
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag


def replace_failed_contents(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace the 'content_or_error' column values with NaN for rows where 'success' is 0.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the news content data.

    Returns:
    - pd.DataFrame: The updated DataFrame with 'content_or_error' values replaced for failed rows.
    """
    df.loc[df['success'] == 0, 'content_or_error'] = np.nan
    return df


def remove_extra_blank_lines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'content_or_error' column by replacing consecutive newline characters with a single one.

    Parameters:
    - df (pd.DataFrame): The DataFrame to process.

    Returns:
    - pd.DataFrame: The DataFrame after processing to remove extra blank lines.
    """
    df['content_or_error'] = df['content_or_error'].str.replace(r'\n+', '\n', regex=True)
    return df


def filter_invalid_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out invalid sentences from the 'content_or_error' column. A sentence is considered invalid if it doesn't contain a verb.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing sentences to be filtered.

    Returns:
    - pd.DataFrame: The DataFrame with invalid sentences removed from the 'content_or_error' column.
    """
    # Ensure necessary NLTK packages are downloaded
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    def is_valid_sentence(text: str) -> bool:
        """
        Determine if a sentence is valid by checking if it contains at least one verb.

        Parameters:
        - text (str): The text to check.

        Returns:
        - bool: True if the sentence is valid, False otherwise.
        """
        sentences = sent_tokenize(text)
        for sentence in sentences:
            words = word_tokenize(sentence)
            tagged = pos_tag(words)
            if not any(tag.startswith('V') for word, tag in tagged):
                return False
        return True

    def filter_content(content: str) -> str:
        """
        Filter content by removing invalid sentences.

        Parameters:
        - content (str): The content to filter.

        Returns:
        - str: The filtered content.
        """
        return '\n'.join(segment for segment in content.split('\n') if is_valid_sentence(segment))

    df['content_or_error'] = df['content_or_error'].apply(lambda x: filter_content(x) if isinstance(x, str) else x)
    return df


def main(year: int) -> None:
    """
    Process news content data by cleaning and filtering it based on specified criteria.

    Parameters:
    - year (int): The year of the data to process.

    Returns:
    - pd.DataFrame: The processed DataFrame.
    """
    csv_path = os.path.join('..', '..', 'data', 'News', 'Initial', 'RawNews', f'{year}.csv')
    csv_saving_path = os.path.join('..', '..', 'data', 'News', 'Initial', 'Cleaned', f'{year}.csv')
    df = pd.read_csv(csv_path, encoding='utf-8')
    df = replace_failed_contents(df)
    df = remove_extra_blank_lines(df)
    df = filter_invalid_sentences(df)
    df.to_csv(csv_saving_path, index=False, encoding='utf-8')


# Process data for years 2000 through 2023 using the main function
for year in range(2000, 2024):
    main(year)

import pandas as pd
import os
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import numpy as np

# Load the tokenizer and model from Hugging Face's Transformers library
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def process_data(file_path: str) -> pd.DataFrame:
    """
    Read a CSV file and preprocess its content by tokenizing a specific column.

    Parameters:
    - file_path (str): The path to the CSV file to be processed.

    Returns:
    - pd.DataFrame: The DataFrame with added columns for tokenized data and token lengths.
    """
    # Read the CSV file
    data = pd.read_csv(file_path, encoding='utf-8')
    print("Token processing started...")

    # Tokenize the content of 'content_or_error' column
    data['tokenized'] = data['content_or_error'].apply(
        lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True) if pd.notna(x) else None)
    data['token_len'] = data['tokenized'].apply(lambda x: len(x) if x is not None else None)

    print("Tokenization complete...")

    return data

def split_token(df: pd.DataFrame, max_token_length: int = 512, overlap: int = 50) -> pd.DataFrame:
    """
    Splits tokens into segments to handle the token limit of the model.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the tokenized data.
    - max_token_length (int): Maximum length of tokens for each segment.
    - overlap (int): Number of tokens that overlap between segments.

    Returns:
    - pd.DataFrame: The DataFrame with tokens split into multiple segments.
    """
    print("Token splitting started...")

    # Determine the maximum token length
    max_token_len = df['token_len'].max()
    # Determine the number of segments needed
    num_segments = max(0, int((max_token_len - 128) / (max_token_length - overlap)) + 1)

    print("Maximum number of tokens:", max_token_len, "...")
    print("Maximum number of segments:", num_segments, "...")

    # Add columns for each segment, initializing all values to None
    for i in range(1, num_segments + 1):
        df[f'token_seg_{i}'] = None

    # Iterate through the DataFrame to split tokens into segments
    for index, row in df.iterrows():
        if not pd.isna(row['token_len']):
            full_tokenized_text = row['tokenized']
            token_len = len(full_tokenized_text)

            start_index = 0
            for i in range(num_segments):
                # Use different minimum length thresholds for the first and subsequent segments
                min_len_threshold = 64 if i == 0 else 128

                # Determine the end index for the segment
                end_index = min(start_index + max_token_length, token_len)

                # Skip segments shorter than the minimum length threshold (for the second segment onwards)
                if i > 0 and (end_index - start_index) < min_len_threshold:
                    break

                # Store the segment of token IDs
                token_segment = full_tokenized_text[start_index:end_index]
                df.at[index, f'token_seg_{i + 1}'] = ' '.join(map(str, token_segment))

                # Update the start index for the next segment
                start_index = end_index - overlap

    print("Token splitting complete...")

    return df

def summarize(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generates summaries for each segment of tokenized text in the DataFrame.

    Parameters:
    - data (pd.DataFrame): The DataFrame with segmented tokenized text.

    Returns:
    - pd.DataFrame: The DataFrame with generated summaries for each segment.
    """
    # Move the model to GPU if available
    model.to(device)

    print("Summary generation started...")

    # Get the maximum number of segments
    max_segments = data.columns.str.extract(r'token_seg_(\d+)').dropna().astype(int).max().item()

    # Generate summaries for each segment
    for i in range(1, max_segments + 1):
        segment_column = f'token_seg_{i}'
        summary_column = f'summary_seg_{i}'
        data[summary_column] = None

        # Convert string representations back into lists of integers before creating tensors
        tokenized_tensors = [
            torch.tensor([int(tok) for tok in str(row[segment_column]).split()])
            for _, row in data.iterrows() if pd.notna(row[segment_column])
        ]
        dataset = torch.utils.data.TensorDataset(torch.nn.utils.rnn.pad_sequence(tokenized_tensors, batch_first=True))
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, pin_memory=True)

        summaries = []
        current_batch = 0

        for batch in loader:
            current_batch += 1
            # Move the data batch to GPU
            batch = tuple(t.to(device, non_blocking=True) for t in batch)

            # Generate the summary
            with torch.no_grad():
                output_ids = model.generate(batch[0], max_length=150, min_length=40, length_penalty=2.0, num_beams=4,
                                            early_stopping=True)

            # Move the generated summary IDs to CPU
            output_ids = output_ids.detach().cpu()

            # Decode the summary and add to the result list
            for ids in output_ids:
                summary = tokenizer.decode(ids, skip_special_tokens=True)
                summaries.append(summary)

            print(f"Processing batch {current_batch}/{len(loader)} on CPU, while GPU is generating summaries.")

        # Add the generated summaries to the DataFrame
        data.loc[data[segment_column].notna(), summary_column] = summaries

    # Concatenate all summary segments
    data['summary'] = data[[f'summary_seg_{i}' for i in range(1, max_segments + 1)]].apply(
        lambda x: ' '.join(x.dropna()), axis=1)

    print("Summary generation complete.")

    return data

def main(year: int) -> pd.DataFrame:
    """
    The main function to process, tokenize, split, and summarize a year's worth of news content.

    Parameters:
    - year (int): The year for which the news content will be summarized.

    Returns:
    - pd.DataFrame: The DataFrame containing the original data and the generated summaries.
    """
    file_path = os.path.join('..', '..', 'data', 'News', 'Initial', 'FurtherCleaned', f'{year}.csv')
    file_saving_path = os.path.join('..', '..', 'data', 'News', 'Summarized', 'Articles', f'{year}.csv')

    data = process_data(file_path)
    data = split_token(data)
    summarized_data = summarize(data)

    print(summarized_data.head())

    # Code to save the file, if needed
    summarized_data.to_csv(file_saving_path, index=False, encoding='utf-8')

    return summarized_data

for year in range(2003, 2024):
    main(year)

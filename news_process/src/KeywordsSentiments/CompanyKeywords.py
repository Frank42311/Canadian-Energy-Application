import pandas as pd
from keybert import KeyBERT
import os


def init_keybert_model() -> KeyBERT:
    """
    Initializes and returns a KeyBERT model with the specified model checkpoint.

    Returns:
        kw_model (KeyBERT): Initialized KeyBERT model.
    """
    print("Initializing KeyBERT model...")
    kw_model = KeyBERT('all-MiniLM-L6-v2')
    print("Model initialized.")
    return kw_model


def extract_keywords_and_update_file(kw_model: KeyBERT, file_path: str, year: int, save_base_path: str):
    """
    Extracts keywords from the summary results of a given CSV file using the KeyBERT model,
    adds a 'year' column, and saves the results to a new file in a specified directory.

    Parameters:
        kw_model (KeyBERT): The KeyBERT model to use for keyword extraction.
        file_path (str): The path to the CSV file to process.
        year (int): The year to add to each row in the CSV.
        save_base_path (str): The base directory where the updated file will be saved.
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return

    # Read specific columns from the file and add a 'year' column
    df = pd.read_csv(file_path, encoding='utf-8', usecols=['company_name', 'sector_name', 'summary_result'])
    df['year'] = year

    new_rows = []  # List to store new rows with extracted keywords
    for index, row in df.iterrows():
        summary_result = row['summary_result'] if pd.notnull(row['summary_result']) else ""
        exclude_terms = [row['company_name'], row['sector_name'], str(year)]

        # Extract keywords and their scores, excluding certain terms and keeping the top 5
        keywords = kw_model.extract_keywords(summary_result, keyphrase_ngram_range=(1, 1), stop_words='english',
                                             use_mmr=True, diversity=0.7, top_n=10)
        filtered_keywords = [kw for kw in keywords if kw[0] not in exclude_terms][:5]

        for keyword, score in filtered_keywords:
            new_row = row.to_dict()
            new_row['keywords'] = keyword
            new_row['keywords_score'] = score
            new_rows.append(new_row)

    # Save the new dataframe to a CSV file
    results_df = pd.DataFrame(new_rows)
    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path)
    new_file_name = os.path.basename(file_path).replace('.csv', '_keywords.csv')
    new_file_path = os.path.join(save_base_path, new_file_name)
    results_df.to_csv(new_file_path, index=False, encoding='utf-8')
    print(f"Processed and saved file: {new_file_path}")


def main():
    kw_model = init_keybert_model()
    base_path = "../../data/News/Summarized/Companies"
    save_base_path = "../../data/News/KeywordsSentiments/CompanyKeywords"

    for year in range(2008, 2024):
        file_name = f"{year}.csv"
        file_path = os.path.join(base_path, file_name)
        extract_keywords_and_update_file(kw_model, file_path, year, save_base_path)


if __name__ == "__main__":
    main()

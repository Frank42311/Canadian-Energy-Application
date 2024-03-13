import pandas as pd
from keybert import KeyBERT
import os


def init_model() -> KeyBERT:
    """
    Initializes the KeyBERT model with the 'all-MiniLM-L6-v2' model.

    Returns:
        KeyBERT: An instance of the KeyBERT model initialized.
    """
    print("Initializing KeyBERT model...")
    kw_model = KeyBERT('all-MiniLM-L6-v2')
    print("Model initialized successfully.")
    return kw_model


def extract_keywords_and_update_file(model: KeyBERT, file_path: str, year: int, save_base_path: str) -> None:
    """
    Reads a file, adds a year column, extracts keywords from a specified column,
    and saves the updated DataFrame to a new file with keywords included.

    Parameters:
        model (KeyBERT): The KeyBERT model used for keyword extraction.
        file_path (str): The path to the input CSV file.
        year (int): The year to add as a new column in the DataFrame.
        save_base_path (str): The base path where the updated file will be saved.
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return

    df = pd.read_csv(file_path, encoding='utf-8')
    df['year'] = year

    if 'sector_summary' in df.columns:
        new_rows = []  # List to store new rows with keywords.

        for index, row in df.iterrows():
            sector_summary = row['sector_summary'] if pd.notnull(row['sector_summary']) else ""

            # Extract up to 7 keywords as candidates.
            keywords = model.extract_keywords(sector_summary, keyphrase_ngram_range=(1, 2), stop_words='english',
                                              use_mmr=True, diversity=0.7, top_n=7)

            # Filter out keywords containing the year or sector name.
            filtered_keywords = [(keyword, score) for keyword, score in keywords if
                                 str(year) not in keyword and row['sector_name'].lower() not in keyword.lower()]

            # Select the top 5 appropriate keywords.
            selected_keywords = filtered_keywords[:5]

            for keyword, score in selected_keywords:
                new_row = row.to_dict()
                new_row['keywords'] = keyword
                new_row['keywords_score'] = score
                new_rows.append(new_row)

        results_df = pd.DataFrame(new_rows)

        if not os.path.exists(save_base_path):
            os.makedirs(save_base_path)
        new_file_name = os.path.basename(file_path).replace('.csv', '_keywords.csv')
        new_file_path = os.path.join(save_base_path, new_file_name)
        results_df.to_csv(new_file_path, index=False, encoding='utf-8')
        print(f"Processed and saved file: {new_file_path}")
    else:
        print(f"'sector_summary' column not found in {file_path}.")


def main() -> None:
    """
    Main function to initialize the model, iterate through files for a range of years,
    extract keywords, and update each file accordingly.
    """
    kw_model = init_model()
    base_path = "../../data/News/Summarized/Sectors"
    save_base_path = "../../data/News/KeywordsSentiments/SectorKeywords"

    for year in range(2008, 2024):
        file_name = f"{year}.csv"
        file_path = os.path.join(base_path, file_name)
        extract_keywords_and_update_file(kw_model, file_path, year, save_base_path)


if __name__ == "__main__":
    main()

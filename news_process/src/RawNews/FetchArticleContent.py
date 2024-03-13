from newspaper import Article
import time
import json
import os
import csv


def process_articles(year: int) -> None:
    """
    Processes news articles from a given year by reading URLs from a JSON file,
    downloading the articles, and saving their content to a CSV file.

    Parameters:
    - year (int): The year for which to process articles. It looks for a JSON file named after this year
      in the 'data/news_urls' directory and outputs the content to a CSV file in the 'data/news_contents' directory.

    Returns:
    - None
    """

    # Construct the file paths for both the input JSON and output CSV using the specified year.
    json_file_path = os.path.join('..', '..', 'data', 'News', 'Initial', 'NewsURLs', f'{year}.json')
    csv_file_path = os.path.join('..', '..', 'data', 'News', 'Initial', 'RawNews', f'{year}.csv')

    # Verify the existence of the JSON file for the given year.
    if not os.path.isfile(json_file_path):
        print(f"The file for the year {year} does not exist at {json_file_path}.")
        return

    # Open and load the JSON data from the specified file.
    with open(json_file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    # Initialize the CSV file for writing the processed article contents or errors.
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        # The CSV header includes the company name, a success flag, and article content or error message.
        csvwriter.writerow(['company_name', 'success', 'content_or_error'])

        # Process each item (company and its associated URLs) in the JSON data.
        for item in json_data:
            company_name = item.get('company_name', 'Unknown')
            urls = item.get('urls', [])
            for url in urls:
                try:
                    # Initialize the Article object, download, and parse the article.
                    article = Article(url)
                    article.download()
                    article.parse()
                    # If successful, write the company name, success flag, and article content to the CSV.
                    csvwriter.writerow([company_name, 1, article.text])
                    print(f"Content from {url} saved.")
                except Exception as e:
                    # On failure, write the company name, failure flag, and error message to the CSV.
                    csvwriter.writerow([company_name, 0, str(e)])
                    print(f"An error occurred while trying to process the URL: {url}")
                # A brief pause to avoid overwhelming the source server.
                time.sleep(2)


# Execute the article processing for each year from 2000 to 2023 if the script is run directly.
if __name__ == '__main__':
    for year in range(2000, 2024):  # Loop through the years 2000 to 2023 inclusive.
        process_articles(year)

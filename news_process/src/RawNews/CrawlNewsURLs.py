from pygooglenews import GoogleNews
import json
import time
from datetime import datetime

# Initialize GoogleNews with English language and Canada as the country
gn = GoogleNews(lang='en', country='CA')

# Function to load company information and sector keywords from JSON files
def load_data() -> tuple:
    """
    Loads company information and sector keywords from specified JSON files.

    Returns:
        tuple: A tuple containing two dictionaries, the first for companies information
               and the second for sector keywords.
    """
    with open("..\..\data\CompanyInformation\JSON\company_info.json", 'r', encoding='utf-8') as file:
        companies_info = json.load(file)

    with open("..\..\data\CompanyInformation\JSON\sector_keywords.json", 'r', encoding='utf-8') as file:
        sector_keywords = json.load(file)

    print(companies_info, sector_keywords)
    return companies_info, sector_keywords

# Function to use pygooglenews to get news related to a company and sector
def get_news(company_name: str, sector_keyword: str, year: int) -> dict:
    """
    Searches for news articles related to a company and its sector within a specified year.

    Parameters:
        company_name (str): The name of the company.
        sector_keyword (str): The keyword associated with the company's sector.
        year (int): The year for which news articles are to be searched.

    Returns:
        dict: A dictionary containing the company name, sector keyword, count of news articles found,
              and URLs of the news articles.
    """
    query = f"{company_name} {sector_keyword}"
    start_date = datetime.strptime(f"{year}-01-01", "%Y-%m-%d").date()
    end_date = datetime.strptime(f"{year}-12-31", "%Y-%m-%d").date()
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    search = gn.search(query=query, from_=start_date_str, to_=end_date_str)

    news_urls = []
    for story in search['entries']:
        news_urls.append(story.link)
        if len(news_urls) >= 10:
            break

    return {
        "company_name": company_name,
        "sector_keyword": sector_keyword,
        "count": len(news_urls),
        "urls": news_urls
    }

# Main crawling function to iterate over years and fetch news for each company and sector
def crawl(year: int) -> str:
    """
    Crawls and saves news URLs related to companies and their sectors for a given year.

    Parameters:
        year (int): The year for which the news articles are to be crawled.

    Returns:
        str: A message indicating the completion of the crawling process.
    """
    companies_info, sector_keywords = load_data()
    all_news_data = []

    for sector, companies in companies_info.items():
        keywords = sector_keywords.get(sector, [''])
        for company in companies:
            for keyword in keywords:
                news_data = get_news(company, keyword, year)
                all_news_data.append(news_data)
                time.sleep(5)  # Pause to avoid hitting request limits
                print(news_data)
    with open(f"..\..\data\\News\Initial\\NewsURLs\{year}.json", 'w', encoding='utf-8') as file:
        json.dump(all_news_data, file, ensure_ascii=False, indent=4)

    return "Crawling completed!"

# Entry point to start the crawling process for a range of years
if __name__ == '__main__':
    for year in range(2000, 2024):  # Iterate from the year 2000 to 2023
        print(crawl(year))

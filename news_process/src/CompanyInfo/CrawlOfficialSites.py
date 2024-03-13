from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import time
import json

# Path to Chrome and Chromedriver
chrome_path = '..\..\chrome-win64\chrome.exe'
driver_path = '..\..\chromedriver\chromedriver.exe'

# Set Chrome options
chrome_options = Options()
chrome_options.binary_location = chrome_path
chrome_options.add_argument('--headless') # Operate in headless mode

# Load JSON data
with open('../../data/CompanyInformation/JSON/company_info.json', 'r', encoding='utf-8') as json_file:
    companies_data = json.load(json_file)

def search_google(value: str) -> str:
    """
    Initializes WebDriver, navigates to Google, and searches for the official site of a given company.

    Parameters:
    - value (str): The company name to search for.

    Returns:
    - str: The URL of the first search result.
    """
    # Initialize WebDriver
    service = Service(executable_path=driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Navigate to Google
        driver.get('https://www.google.com')

        # Search for the value
        search_box = driver.find_element("name", 'q')
        search_box.send_keys(value + ' official site')
        search_box.send_keys(Keys.RETURN)

        # Wait for the page to load
        time.sleep(1)

        # Fetch the first link
        first_result = driver.find_element("css selector", 'div#search a')
        link = first_result.get_attribute('href')

        return link
    finally:
        driver.quit()

def extract_companies(sector: str = 'ALL') -> dict:
    """
    Filters companies by sector from a loaded dataset and searches Google for their official websites.

    Parameters:
    - sector (str): The sector to filter companies by. Defaults to 'ALL' for no filtering.

    Returns:
    - dict: A dictionary mapping company names to their official website URLs.
    """
    filtered_companies = companies_data[sector] if sector in companies_data and sector != 'ALL' else companies_data

    results = {}
    for sector, companies in filtered_companies.items():
        for company in companies:
            link = search_google(company)
            results[company] = link

    output_path = '../../data/CompanyInformation/OfficialSites/company_official_site.json'
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(results, file)

    return results

if __name__ == '__main__':
    # Example of how to call the extract_companies function
    # You can change 'IT' to any other sector you have in your data or use 'ALL' to process all sectors
    sector = 'ALL'
    company_websites = extract_companies(sector)
    print(f"Extracted websites for sector '{sector}':")
    for company, website in company_websites.items():
        print(f"{company}: {website}")

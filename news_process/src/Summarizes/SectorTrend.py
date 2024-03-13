import pandas as pd
from openai import OpenAI


# Initialize the OpenAI API client with an API key from a file
def initialize_openai_client(api_key_file_path: str) -> OpenAI:
    """
    Initializes the OpenAI client with an API key.

    Args:
        api_key_file_path (str): The file path to the API key.

    Returns:
        OpenAI: An initialized OpenAI client.
    """
    with open(api_key_file_path, 'r') as file:
        api_key = file.read().strip()
    client = OpenAI(api_key=api_key)
    return client


# Function to interact with GPT and handle possible errors
def get_gpt_response(client: OpenAI, prompt: str) -> str:
    """
    Sends a prompt to GPT and returns its response, handling any errors.

    Args:
        client (OpenAI): The OpenAI client initialized with an API key.
        prompt (str): The prompt to send to GPT.

    Returns:
        str: The response from GPT or an error message.
    """
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="gpt-4",  # Note: Model names may update, please adjust based on current models available.
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return str(e)


# Main function to generate sector analysis prompts, get GPT responses, and save the results
def generate_sector_prompts_and_save_results(api_key_file_path: str) -> None:
    """
    Generates analysis prompts for various sectors over years, gets responses from GPT, and saves the results.

    Args:
        api_key_file_path (str): The file path to the API key for initializing the OpenAI client.
    """
    # Initialize OpenAI client
    client = initialize_openai_client(api_key_file_path)

    sector_dfs = []  # List to store DataFrames for each year

    # Loop through the years of interest to process sector data
    for year in range(2008, 2024):
        sector_file_path = f"../../data/News/Summarized/Sectors/{year}.csv"
        sector_df = pd.read_csv(sector_file_path)
        sector_df = sector_df[['sector_name', 'sector_summary', 'year']]
        sector_df['year_summary'] = "Summary in " + sector_df['year'].astype(str) + ": " + sector_df['sector_summary']
        sector_df = sector_df.drop_duplicates(subset=['sector_name'])
        sector_dfs.append(sector_df)

    combined_sector_df = pd.concat(sector_dfs, ignore_index=True)
    results_df = pd.DataFrame(columns=['sector_name', 'sector_trend'])
    sector_grouped = combined_sector_df.groupby('sector_name')

    for sector_name, group in sector_grouped:
        group_sorted = group.sort_values('year')
        prompt_intro = f"Provide a comprehensive analysis on {sector_name} sector trends. Include developments, challenges, and achievements. Focus on technology, market dynamics, regulatory changes, and milestones. Highlight the main challenges and forces hindering the sector's development, offering a balanced view.\n\n"
        prompt_details = "\n".join(
            [f"Summary in {row['year']}: {row['sector_summary']}" for _, row in group_sorted.iterrows()])
        full_prompt = prompt_intro + prompt_details
        gpt_response = get_gpt_response(client, full_prompt)
        new_row = {'sector_name': sector_name, 'sector_trend': gpt_response}
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

    results_df.to_csv("../../data/News/Summarized/Trends/sector_trend.csv", index=False, encoding='utf-8')
    print("Results saved to ../../data/News/Summarized/Trends/sector_trend.csv")

# Example call to the main function
generate_sector_prompts_and_save_results('../../gpt_api/api_key.txt')

import pandas as pd

def process_data(file_path_template, years_range, group_columns, agg_operations, non_agg_columns, rename_columns):
    """
    Processes data by reading CSV files, aggregating, and merging based on specified parameters.

    :param file_path_template: A string template for the file paths to read, with placeholders for years.
    :param years_range: A range object specifying the years of data to process.
    :param group_columns: A list of columns to group the data by.
    :param agg_operations: A dictionary specifying how to aggregate each column after grouping.
    :param non_agg_columns: A list of columns that do not require aggregation.
    :param rename_columns: A dictionary specifying columns to rename.
    :return: A DataFrame that has been processed based on the above parameters.
    """
    dfs = []
    for year in years_range:
        file_path = file_path_template.format(year=year)
        df = pd.read_csv(file_path)
        dfs.append(df)
    df_concat = pd.concat(dfs, ignore_index=True)

    # Perform aggregation based on original column names
    df_agg = df_concat.groupby(group_columns).agg(agg_operations).reset_index()

    # Ensure non_agg_columns does not include any of the group_columns to avoid conflicts
    non_agg_columns_filtered = [col for col in non_agg_columns if col not in group_columns]

    # Handle non-aggregated columns if there are any to include
    if non_agg_columns_filtered:
        df_first = df_concat.groupby(group_columns)[non_agg_columns_filtered].first().reset_index()
        # Merge aggregated and non-aggregated data
        df_merged = pd.merge(df_agg, df_first, on=group_columns)
    else:
        df_merged = df_agg

    # Rename columns after aggregation and merging
    df_merged.rename(columns=rename_columns, inplace=True)

    return df_merged

def merge_data(company_df, sector_df, sector_trend_path):
    """
    Merges processed company and sector DataFrames with sector trend data, preparing a final DataFrame.

    :param company_df: The DataFrame containing processed company data.
    :param sector_df: The DataFrame containing processed sector data.
    :param sector_trend_path: The file path to the sector trend data CSV.
    :return: A merged DataFrame containing company and sector data along with sector trends.
    """
    merged_df = pd.merge(company_df, sector_df, how='left', on=['year', 'sector_name'])
    sector_trend_df = pd.read_csv(sector_trend_path)
    final_df = pd.merge(merged_df, sector_trend_df[['sector_name', 'sector_trend']], on='sector_name', how='left')
    return final_df

# Define file path templates, years range, group and aggregation settings for sectors and companies
years_range = range(2008, 2024)
sector_file_path_template = "../../data/News/KeywordsSentiments/SectorSentiments/{year}_keywords.csv"
company_file_path_template = "../../data/News/KeywordsSentiments/CompanySentiments/{year}_keywords.csv"
sector_group_columns = ['year', 'sector_name']
company_group_columns = ['year', 'company_name']
agg_operations = {
    'keywords': lambda x: '; '.join(x),
    'keywords_score': lambda x: '; '.join(map(str, x)),
    'keywords_sentiment': lambda x: '; '.join(map(str, x))
}
sector_rename_columns = {
    'keywords': 'sector_keywords',
    'keywords_score': 'sector_keywords_score',
    'keywords_sentiment': 'sector_keywords_sentiment'
}
company_rename_columns = {
    'keywords': 'company_keywords',
    'keywords_score': 'company_keywords_score',
    'keywords_sentiment': 'company_keywords_sentiment',
    'summary_result': 'company_summary'
}

# Process sector and company data
sector_df = process_data(sector_file_path_template, years_range, sector_group_columns, agg_operations, ['sector_name'], sector_rename_columns)
company_df = process_data(company_file_path_template, years_range, company_group_columns, agg_operations, ['company_name', 'sector_name'], company_rename_columns)

# Merge processed data into a final DataFrame
final_df_with_trend = merge_data(company_df, sector_df, "../../data/News/Summarized/Trends/sector_trend.csv")

# Save the final DataFrame to a CSV file
final_df_with_trend.to_csv("../../data/Final/final.csv", index=False, encoding='utf-8')

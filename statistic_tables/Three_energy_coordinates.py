import pandas as pd
import numpy as np

def select_and_process_columns(df, prefix, input_multiplier, output_multiplier) -> pd.DataFrame:
    """
    Selects specified columns, processes the 'Supply and disposition' column by adding a prefix,
    replaces NaN values in 'VALUE' column with 0, and calculates input and output energy in MJ.

    Parameters:
    - df: DataFrame to process.
    - prefix: String prefix to add to the 'Supply and disposition' column.
    - input_multiplier: Multiplier to calculate input MJ.
    - output_multiplier: Multiplier to calculate output MJ.

    Returns:
    - Processed DataFrame.
    """
    # Select and preprocess specified columns
    df = df[['REF_DATE', 'GEO', 'Supply and disposition', 'UOM', 'VALUE']]
    df.loc[:, 'Supply and disposition'] = prefix + ' ' + df['Supply and disposition']
    df.loc[:, 'VALUE'] = df['VALUE'].replace(np.nan, 0)

    new_rows = pd.DataFrame(columns=df.columns)
    input_conditions = {
        'Gas': ['Gross withdrawals', 'Marketable production', 'Imports'],
        'Crude': ['Heavy crude oil', 'Light and medium crude oil', 'Non-upgraded production of crude bitumen',
                  'In-Situ crude bitumen production', 'Mined crude bitumen production',
                  'Crude bitumen sent for further processing', 'Synthetic crude oil production', 'Condensate',
                  'Pentanes plus', 'Imports of crude oil and equivalent products by refineries',
                  'Imports of crude oil and equivalent products by other'],
        'Coal': ['Coal coke, received', 'Coal coke, production', 'Coal, received']
    }
    output_conditions = {
        'Gas': ['Commercial consumption', 'Industrial consumption', 'Residential consumption', 'Exports'],
        'Crude': ['Light and medium crude oil used as an input in refineries', 'Heavy crude oil used as an input in refineries',
                  'Crude bitumen used as an input in refineries', 'Synthetic crude oil used as an input in refineries',
                  'Export to the United States by pipelines', 'Export to the United States by other means', 'Export to other countries'],
        'Coal': ['Coal coke, used in blast furnaces', 'Coal coke, sold for industrial use',
                 'Coal coke, used in associated works', 'Coal coke, all other sales and/or uses',
                 'Coal coke, sold for export', 'Coal, charged to ovens', 'Coal, sold or used for other purposes']
    }

    # Add energy calculations
    for (ref_date, geo), group in df.groupby(['REF_DATE', 'GEO']):
        input_value = group.loc[group['Supply and disposition'].str.contains('|'.join(input_conditions[prefix[:-1]])), 'VALUE'].astype(float).sum() * input_multiplier
        output_value = group.loc[group['Supply and disposition'].str.contains('|'.join(output_conditions[prefix[:-1]])), 'VALUE'].astype(float).sum() * output_multiplier

        new_input_row = pd.DataFrame([{'REF_DATE': ref_date, 'GEO': geo, 'Supply and disposition': f'{prefix}Input MJ', 'UOM': 'MJ', 'VALUE': input_value}])
        new_output_row = pd.DataFrame([{'REF_DATE': ref_date, 'GEO': geo, 'Supply and disposition': f'{prefix}Output MJ', 'UOM': 'MJ', 'VALUE': output_value}])

        new_rows = pd.concat([new_rows, new_input_row, new_output_row], ignore_index=True)

    df = pd.concat([df, new_rows], ignore_index=True).sort_values(by=['REF_DATE', 'GEO'])
    return df

def merge_and_adjust_coordinates(df, coor_df, radius_offset=2) -> pd.DataFrame:
    """
    Merges energy data with geographical coordinates, adjusts coordinates based on energy type to prevent overlap,
    and handles regional grouping.

    Parameters:
    - df: The energy DataFrame to adjust.
    - coor_df: DataFrame containing geographical coordinates.
    - radius_offset: Radius to offset the latitude and longitude for different energy types.

    Returns:
    - Adjusted DataFrame with coordinates.
    """
    merged_df = pd.merge(df, coor_df[['GEO', 'Latitude', 'Longitude']], on='GEO', how='left')

    # Handle regional grouping by modifying 'GEO' values
    merged_df['GEO2'] = merged_df['GEO'].replace({
        'Yukon': 'Northern Territories',
        'Northwest Territories': 'Northern Territories',
        'Nunavut': 'Northern Territories',
        'Prince Edward Island': 'Atlantic Provinces',
        'Nova Scotia': 'Atlantic Provinces',
        'New Brunswick': 'Atlantic Provinces'
    })

    # Calculate average coordinates for grouped regions
    region_averages = merged_df.groupby('GEO2')['Latitude', 'Longitude'].mean().reset_index()
    merged_df = pd.merge(merged_df, region_averages, on='GEO2', suffixes=('', '_avg'))
    merged_df['Latitude'] = merged_df['Latitude_avg']
    merged_df['Longitude'] = merged_df['Longitude_avg']

    # Determine the offset for each energy type
    energy_types = df['Supply and disposition'].unique()
    angles = np.linspace(0, 2 * np.pi, len(energy_types), endpoint=False)
    type_to_angle = dict(zip(energy_types, angles))

    for energy_type in energy_types:
        angle = type_to_angle[energy_type]
        delta_lat = radius_offset * np.sin(angle)
        delta_lon = radius_offset * np.cos(angle)
        condition = merged_df['Supply and disposition'] == energy_type
        merged_df.loc[condition, 'Latitude'] += delta_lat
        merged_df.loc[condition, 'Longitude'] += delta_lon

    return merged_df.drop(['Latitude_avg', 'Longitude_avg'], axis=1)

def main() -> None:
    # Load the data
    coal_df = pd.read_csv('data/statistical_tables_updated/coal_coke.csv', encoding='utf-8', low_memory=False)
    crude_df = pd.read_csv('data/statistical_tables_updated/crude_oil.csv', encoding='utf-8', low_memory=False)
    gas_df = pd.read_csv('data/statistical_tables_updated/natural_gas.csv', encoding='utf-8', low_memory=False)
    GEO_coor_df = pd.read_csv("data/statistical_tables/GEO_coordinates.csv", encoding='utf-8', low_memory=False)

    # Process the dataframes
    gas_df = select_and_process_columns(gas_df, 'Gas', 38900, 38900)
    crude_df = select_and_process_columns(crude_df, 'Crude', 35700, 35700)
    coal_df = select_and_process_columns(coal_df, 'Coal', 29, 29)  # Coal has a different multiplier for input and output, adjust as needed

    # Combine the processed DataFrames
    final_df = pd.concat([gas_df, crude_df, coal_df], ignore_index=True)

    # Merge with coordinates and adjust
    adjusted_df = merge_and_adjust_coordinates(final_df, GEO_coor_df)

    # Export the final DataFrame
    adjusted_df.to_csv('data/statistical_tables_updated/three_energy_coor.csv', encoding='utf-8', index=False)
    print(adjusted_df.head(50))

if __name__ == "__main__":
    main()

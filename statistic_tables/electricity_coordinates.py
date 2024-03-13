from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple

def load_dataframes(generation_path, coordinates_path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the electric power generation data and geographic coordinates data into pandas DataFrames.
    """
    ele_gen_df = pd.read_csv(generation_path, low_memory=False)
    GEO_coor_df = pd.read_csv(coordinates_path)
    return ele_gen_df, GEO_coor_df

def merge_dataframes(ele_gen_df, GEO_coor_df) -> pd.DataFrame:
    """
    Merges the electric power generation dataframe with the geographic coordinates dataframe on the 'GEO' column.
    Adds 'longitude' and 'latitude' columns to the electric power generation dataframe.
    """
    merged_df = pd.merge(ele_gen_df, GEO_coor_df[['GEO', 'Latitude', 'Longitude']], on='GEO', how='left')
    return merged_df

def create_GEO2_column(merged_df) -> pd.DataFrame:
    """
    Creates a new column 'GEO2' with specified replacements to update regional names.
    """
    replacements = {
        'Yukon': 'Northern Territories',
        'Northwest Territories': 'Northern Territories',
        'Nunavut': 'Northern Territories',
        'Prince Edward Island': 'Atlantic Provinces',
        'Nova Scotia': 'Atlantic Provinces',
        'New Brunswick': 'Atlantic Provinces'
    }
    merged_df['GEO2'] = merged_df['GEO'].replace(replacements)
    return merged_df

def update_average_coordinates(merged_df) -> pd.DataFrame:
    """
    Updates the latitude and longitude of 'Northern Territories' and 'Atlantic Provinces' with their average values.
    """
    northern_avg = merged_df.loc[merged_df['GEO2'] == 'Northern Territories', ['Latitude', 'Longitude']].mean()
    atlantic_avg = merged_df.loc[merged_df['GEO2'] == 'Atlantic Provinces', ['Latitude', 'Longitude']].mean()

    merged_df.loc[merged_df['GEO2'] == 'Northern Territories', ['Latitude', 'Longitude']] = northern_avg.values
    merged_df.loc[merged_df['GEO2'] == 'Atlantic Provinces', ['Latitude', 'Longitude']] = atlantic_avg.values
    return merged_df

def offset_coordinates_by_generation_type(merged_df, radius_offset=2) -> pd.DataFrame:
    """
    Offsets the latitude and longitude coordinates based on the generation type and geographical location.
    """
    number_of_generation_types = len(merged_df['Type of electricity generation'].unique())
    angles = np.linspace(0, 2 * np.pi, number_of_generation_types, endpoint=False)

    generation_type_to_angle = dict(zip(merged_df['Type of electricity generation'].unique(), angles))

    for geo in merged_df['GEO2'].unique():
        for gen_type in merged_df['Type of electricity generation'].unique():
            center_latitude = merged_df.loc[(merged_df['GEO2'] == geo) & (
                        merged_df['Type of electricity generation'] == gen_type), 'Latitude'].mean()
            center_longitude = merged_df.loc[(merged_df['GEO2'] == geo) & (
                        merged_df['Type of electricity generation'] == gen_type), 'Longitude'].mean()

            angle = generation_type_to_angle[gen_type]
            delta_lat = radius_offset * np.sin(angle)
            delta_lon = radius_offset * np.cos(angle)

            condition = (merged_df['GEO2'] == geo) & (merged_df['Type of electricity generation'] == gen_type)
            merged_df.loc[condition, 'Latitude'] += delta_lat
            merged_df.loc[condition, 'Longitude'] += delta_lon

    return merged_df

def main() -> None:
    # Define paths to the datasets
    generation_path = "data/statistical_tables_updated/electric_power_generation.csv"
    coordinates_path = "data/statistical_tables/GEO_coordinates.csv"

    # Load the data
    ele_gen_df, GEO_coor_df = load_dataframes(generation_path, coordinates_path)

    # Merge dataframes
    merged_df = merge_dataframes(ele_gen_df, GEO_coor_df)

    # Update GEO column
    merged_df = create_GEO2_column(merged_df)

    # Update coordinates to averages for specific regions
    merged_df = update_average_coordinates(merged_df)

    # Offset coordinates based on generation type
    merged_df = offset_coordinates_by_generation_type(merged_df)

    # Save the updated dataframe
    merged_df.to_csv(r'data/statistical_tables_updated/electric_power_generation_coor.csv', index=False,
                     encoding='utf-8')
    print("Data processing complete and file saved.")

if __name__ == "__main__":
    main()

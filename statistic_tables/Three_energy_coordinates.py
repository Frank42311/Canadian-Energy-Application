import pandas as pd
import numpy as np
from typing import Tuple

def process_energy_df(df: pd.DataFrame, energy_type: str) -> pd.DataFrame:
    """
    Process the energy DataFrame by selecting specific columns, prefixing the 'Supply and disposition' values,
    replacing NaN in 'VALUE' with 0, and adding new rows for energy input and output in MJ.

    Args:
    - df: DataFrame containing energy data.
    - energy_type: The type of energy ('Gas', 'Crude', or 'Coal').

    Returns:
    - DataFrame after processing.
    """
    # Keep specified columns
    df = df[['REF_DATE', 'GEO', 'Supply and disposition', 'UOM', 'VALUE']]
    # Prefix 'Supply and disposition' values with energy type
    df['Supply and disposition'] = f'{energy_type} ' + df['Supply and disposition']
    # Replace NaN in 'VALUE' with 0
    df['VALUE'] = df['VALUE'].replace(np.nan, 0)

    # Define an empty DataFrame for new rows
    new_rows = pd.DataFrame(columns=df.columns)

    # Constants for energy conversion
    conversion_factors = {
        'Gas': 38900,
        'Crude': 35700,
        'Coal': {'Coal coke': 29, 'Coal': 25}
    }

    # Determine conversion factor or factors
    if energy_type == 'Coal':
        coal_coke_factor, coal_factor = conversion_factors[energy_type].values()
    else:
        conversion_factor = conversion_factors[energy_type]

    # Process DataFrame based on energy type
    for (ref_date, geo), group in df.groupby(['REF_DATE', 'GEO']):
        if energy_type in ['Gas', 'Crude']:
            # Calculate input and output values for Gas and Crude
            input_output_values = calculate_energy_io(group, energy_type, conversion_factor)
        elif energy_type == 'Coal':
            # Calculate input and output values for Coal
            input_output_values = calculate_coal_io(group, coal_coke_factor, coal_factor)

        # Add new rows for input and output values
        new_input_row, new_output_row = create_io_rows(ref_date, geo, energy_type, input_output_values)
        new_rows = pd.concat([new_rows, new_input_row, new_output_row], ignore_index=True)

    # Merge new rows back to the original DataFrame
    df = pd.concat([df, new_rows], ignore_index=True)
    # Sort the DataFrame as needed
    df = df.sort_values(by=['REF_DATE', 'GEO'])

    return df

def calculate_energy_io(group: pd.DataFrame, energy_type: str, conversion_factor: float) -> Tuple[float, float]:
    """
    Calculate input and output energy values for Gas and Crude.

    Args:
    - group: Grouped DataFrame for a specific REF_DATE and GEO.
    - energy_type: The type of energy ('Gas' or 'Crude').
    - conversion_factor: The factor to convert values to MJ.

    Returns:
    - Tuple of (input_value, output_value).
    """
    if energy_type == 'Gas':
        input_conditions = ['Gas Gross withdrawals', 'Gas Marketable production', 'Gas Imports']
        output_conditions = ['Gas Commercial consumption', 'Gas Industrial consumption',
                             'Gas Residential consumption', 'Gas Exports']
    elif energy_type == 'Crude':
        input_conditions = ['Crude Heavy crude oil', 'Crude Light and medium crude oil',
                            'Crude Non-upgraded production of crude bitumen', 'Crude In-Situ crude bitumen production',
                            'Crude Mined crude bitumen production', 'Crude Crude bitumen sent for further processing',
                            'Crude Synthetic crude oil production', 'Crude Condensate',
                            'Crude Pentanes plus', 'Crude Imports of crude oil and equivalent products by refineries',
                            'Crude Imports of crude oil and equivalent products by other']
        output_conditions = ['Crude Light and medium crude oil used as an input in refineries',
                             'Crude Heavy crude oil used as an input in refineries',
                             'Crude Crude bitumen used as an input in refineries',
                             'Crude Synthetic crude oil used as an input in refineries',
                             'Crude Export to the United States by pipelines',
                             'Crude Export to the United States by other means',
                             'Crude Export to other countries']

    input_value = group.loc[group['Supply and disposition'].isin(input_conditions), 'VALUE'].astype(
        float).sum() * conversion_factor
    output_value = group.loc[group['Supply and disposition'].isin(output_conditions), 'VALUE'].astype(
        float).sum() * conversion_factor

    return input_value, output_value

def calculate_coal_io(group: pd.DataFrame, coal_coke_factor: float, coal_factor: float) -> Tuple[float, float]:
    """
    Calculate input and output energy values for Coal.

    Args:
    - group: Grouped DataFrame for a specific REF_DATE and GEO.
    - coal_coke_factor: Conversion factor for coal coke to MJ.
    - coal_factor: Conversion factor for coal to MJ.

    Returns:
    - Tuple of (input_value, output_value).
    """
    # Define conditions for coal and coal coke for both input and output
    input_conditions_coke = ['Coal Coal coke, received', 'Coal Coal coke, production']
    output_conditions_coke = ['Coal Coal coke, used in blast furnaces', 'Coal Coal coke, sold for industrial use',
                              'Coal Coal coke, used in associated works', 'Coal Coal coke, all other sales and/or uses',
                              'Coal Coal coke, sold for export']
    input_conditions_coal = ['Coal Coal, received']
    output_conditions_coal = ['Coal Coal, charged to ovens', 'Coal Coal, sold or used for other purposes']

    # Calculate input and output values separately for coal coke and coal
    input_value_coke = group.loc[group['Supply and disposition'].isin(input_conditions_coke), 'VALUE'].astype(
        float).sum() * coal_coke_factor
    output_value_coke = group.loc[group['Supply and disposition'].isin(output_conditions_coke), 'VALUE'].astype(
        float).sum() * coal_coke_factor
    input_value_coal = group.loc[group['Supply and disposition'].isin(input_conditions_coal), 'VALUE'].astype(
        float).sum() * coal_factor
    output_value_coal = group.loc[group['Supply and disposition'].isin(output_conditions_coal), 'VALUE'].astype(
        float).sum() * coal_factor

    # Sum the input and output values for coal coke and coal
    input_value = input_value_coke + input_value_coal
    output_value = output_value_coke + output_value_coal

    return input_value, output_value

def create_io_rows(ref_date: str, geo: str, energy_type: str, input_output_values: Tuple[float, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create new DataFrame rows for energy input and output values.

    Args:
    - ref_date: The reference date.
    - geo: The geographical location.
    - energy_type: The type of energy ('Gas', 'Crude', or 'Coal').
    - input_output_values: Tuple containing input and output energy values.

    Returns:
    - Tuple of DataFrames (new_input_row, new_output_row).
    """
    input_value, output_value = input_output_values
    new_input_row = pd.DataFrame([{'REF_DATE': ref_date, 'GEO': geo,
                                   'Supply and disposition': f'{energy_type} Input MJ', 'UOM': 'MJ',
                                   'VALUE': input_value}])
    new_output_row = pd.DataFrame([{'REF_DATE': ref_date, 'GEO': geo,
                                    'Supply and disposition': f'{energy_type} Output MJ', 'UOM': 'MJ',
                                    'VALUE': output_value}])
    return new_input_row, new_output_row

def merge_and_adjust_coordinates(final_df: pd.DataFrame, GEO_coor_df: pd.DataFrame, radius_offset: float = 2) -> pd.DataFrame:
    """
    Merge the final DataFrame with GEO coordinates and adjust locations for visualization.

    Args:
    - final_df: The final DataFrame containing all energy data.
    - GEO_coor_df: DataFrame containing geographical coordinates.
    - radius_offset: The radius offset for adjusting coordinates.

    Returns:
    - DataFrame after merging and adjusting coordinates.
    """
    # Merge dataframes to add latitude and longitude columns
    merged_df = pd.merge(final_df, GEO_coor_df[['GEO', 'Latitude', 'Longitude']], on='GEO', how='left')

    # Create a new 'GEO2' column for grouping territories and provinces
    merged_df['GEO2'] = merged_df['GEO'].replace({
        'Yukon': 'Northern Territories',
        'Northwest Territories': 'Northern Territories',
        'Nunavut': 'Northern Territories',
        'Prince Edward Island': 'Atlantic Provinces',
        'Nova Scotia': 'Atlantic Provinces',
        'New Brunswick': 'Atlantic Provinces'
    })

    # Calculate average coordinates for grouped locations
    northern_avg = merged_df.loc[merged_df['GEO2'] == 'Northern Territories', ['Latitude', 'Longitude']].mean()
    atlantic_avg = merged_df.loc[merged_df['GEO2'] == 'Atlantic Provinces', ['Latitude', 'Longitude']].mean()

    # Update coordinates for grouped locations
    merged_df.loc[merged_df['GEO2'] == 'Northern Territories', ['Latitude', 'Longitude']] = northern_avg.values
    merged_df.loc[merged_df['GEO2'] == 'Atlantic Provinces', ['Latitude', 'Longitude']] = atlantic_avg.values

    # Offset coordinates based on energy type
    offset_coordinates(merged_df, radius_offset)

    return merged_df

def offset_coordinates(merged_df: pd.DataFrame, radius_offset: float) -> None:
    """
    Offset the coordinates of energy types for clearer visualization.

    Args:
    - merged_df: DataFrame with merged energy and geographic data.
    - radius_offset: The radius offset for adjusting coordinates.
    """
    # Determine MJ types and corresponding angles
    number_of_UOM_types = 6
    angles = np.linspace(0, 2 * np.pi, number_of_UOM_types, endpoint=False)
    MJ_types = ['Gas Input MJ', 'Gas Output MJ', 'Crude Input MJ', 'Crude Output MJ', 'Coal Input MJ', 'Coal Output MJ']
    MJ_type_to_angle = dict(zip(MJ_types, angles))

    # Offset location for each province and MJ type
    for geo in merged_df['GEO2'].unique():
        for MJ_type in MJ_types:
            condition = (merged_df['GEO2'] == geo) & (merged_df['Supply and disposition'] == MJ_type)
            if condition.any():
                angle = MJ_type_to_angle[MJ_type]
                delta_lat = radius_offset * np.sin(angle)
                delta_lon = radius_offset * np.cos(angle)
                merged_df.loc[condition, 'Latitude'] += delta_lat
                merged_df.loc[condition, 'Longitude'] += delta_lon

def main() -> None:
    # Load data
    coal_df = pd.read_csv('data/statistical_tables_updated/coal_coke.csv', encoding='utf-8', low_memory=False)
    crude_df = pd.read_csv('data/statistical_tables_updated/crude_oil.csv', encoding='utf-8', low_memory=False)
    gas_df = pd.read_csv('data/statistical_tables_updated/natural_gas.csv', encoding='utf-8', low_memory=False)
    GEO_coor_df = pd.read_csv("data/statistical_tables/GEO_coordinates.csv", encoding='utf-8', low_memory=False)

    # Process DataFrames
    gas_df_processed = process_energy_df(gas_df, 'Gas')
    crude_df_processed = process_energy_df(crude_df, 'Crude')
    coal_df_processed = process_energy_df(coal_df, 'Coal')

    # Merge processed DataFrames
    final_df = pd.concat([gas_df_processed, crude_df_processed, coal_df_processed], ignore_index=True)

    # Merge with GEO coordinates and adjust
    adjusted_df = merge_and_adjust_coordinates(final_df, GEO_coor_df)

    # Save the adjusted DataFrame
    adjusted_df.to_csv('data/statistical_tables_updated/three_energy_coor.csv', encoding='utf-8', index=False)
    print("Data processing and adjustment complete.")

if __name__ == "__main__":
    main()

import pandas as pd
from pathlib import Path
import pmdarima as pm

def prepare_and_predict(file_path: Path, save_path: Path, group_columns: list, filter_conditions=None, start_date='2016-01-01',
                        end_date='2023-10-31') -> None:
    """
    General function for data preparation and prediction.

    Parameters:
    - file_path: Path to the data file.
    - save_path: Path to save the result file.
    - group_columns: List of column names to group the data.
    - filter_conditions: Conditions for filtering data, should be a function that takes a dataframe and returns a boolean Series.
    - start_date: Start date for filtering the data.
    - end_date: End date for filtering the data.
    """

    # Read data
    data = pd.read_csv(file_path, encoding='utf-8')
    data['REF_DATE'] = pd.to_datetime(data['REF_DATE'])
    data['VALUE'] = data['VALUE'].astype(float)  # Ensure VALUE is float type

    if filter_conditions:
        data = data[filter_conditions(data)]

    # Filter data by date
    filtered_data = data[(data['REF_DATE'] >= start_date) & (data['REF_DATE'] <= end_date)]

    # Aggregate data
    grouped_data = filtered_data.groupby(group_columns + [pd.Grouper(key='REF_DATE', freq='M')])[
        'VALUE'].sum().reset_index()

    # Prepare container for prediction results
    predictions = []

    # Perform predictions
    for group_values, group in grouped_data.groupby(group_columns):
        group.set_index('REF_DATE', inplace=True)
        group.index = pd.DatetimeIndex(group.index).to_period('M')

        if len(group) < 24:  # At least 2 years of monthly data is required to build the model
            continue

        try:
            auto_model = pm.auto_arima(group['VALUE'], seasonal=True, m=12, stepwise=True,
                                       suppress_warnings=True, error_action='ignore', maxiter=100)
            forecast = auto_model.predict(n_periods=2)
            forecast_index = pd.date_range(start='2023-11-01', periods=2, freq='MS')

            for i, value in enumerate(forecast):
                prediction = dict(zip(group_columns, group_values))
                prediction.update({'REF_DATE': forecast_index[i], 'VALUE': value})
                predictions.append(prediction)

        except Exception as e:
            print(f"Model building failed: {e}")

    # Process prediction results
    if predictions:
        predictions_df = pd.DataFrame(predictions).sort_values(by=['REF_DATE'] + group_columns)
        final_data = pd.concat([data, predictions_df])
        final_data.to_csv(save_path, index=False, encoding='utf-8')
    else:
        print("Insufficient data for predictions.")

# Define file paths and parameters
paths_and_params = [
    ("data/statistical_tables/coal_coke.csv", "data/statistical_tables_updated/coal_coke.csv",
     ['GEO', 'Supply and disposition']),
    ("data/statistical_tables/electric_power_generation.csv",
     "data/statistical_tables_updated/electric_power_generation.csv",
     ['GEO', 'Class of electricity producer', 'Type of electricity generation']),
    ("data/statistical_tables/natural_gas.csv", "data/statistical_tables_updated/natural_gas.csv",
     ['GEO', 'Supply and disposition'], lambda x: x['Unit of measure'] == 'Cubic metres'),
    ("data/statistical_tables/crude_oil.csv", "data/statistical_tables_updated/crude_oil.csv",
     ['GEO', 'Supply and disposition'], lambda x: x['Units of measure'] == 'Cubic metres'),
]

# Execute predictions
for file_path, save_path, group_columns, *filter_condition in paths_and_params:
    prepare_and_predict(Path(file_path), Path(save_path), group_columns, *filter_condition)

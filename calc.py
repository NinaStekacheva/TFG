import pandas as pd
from sklearn.metrics import mean_squared_error

# Read the CSV file
data = pd.read_csv('auto_arima_results.csv')


# Calculate RMSE average grouped by Pollutant

# Exclude specific rows
excluded_rows = ((data['Pollutant'] == 'O3') & (data['Station'].isin(['station_Barcelona (Sants)', 'station_Barcelona (Poblenou)'])) | ((data['Pollutant'] == 'PM10') & (data['Station'].isin(['station_Barcelona (Sants)', 'station_Barcelona (Ciutadella)'])))| ((data['Pollutant'] == 'NOX') & (data['Station'].isin(['station_Barcelona (Eixample)']))))
filtered_data = data[~excluded_rows]

# Calculate RMSE average grouped by Pollutant
average_rmse = filtered_data.groupby(['Pollutant'])['MAPE'].mean()

# Print the results
print(average_rmse.round(2))
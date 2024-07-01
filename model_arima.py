
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
from pmdarima import auto_arima
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#################################################################################################
#################################################################################################
## This script performs ARIMA-based time series forecasting on urban air quality data for Barcelona. 
## It processes weekly aggregated data from multiple stations, automatically selects the best ARIMA model parameters,
## evaluates model performance with several accuracy metrics, visualizes predictions vs. actual values, 
## and saves the results in a CSV file for further analysis.
#################################################################################################
#################################################################################################

imputed_data = pd.read_csv('imputed_data.csv')
imputed_data.head()
#imputed_data['traffic'] = np.where((imputed_data['station_Barcelona (Eixample)'] == 1) | (imputed_data['station_Barcelona (Gràcia - Sant Gervasi)'] == 1), 1, 0)
#imputed_data['background'] = np.where((imputed_data['station_Barcelona (Ciutadella)'] == 1) | (imputed_data['station_Barcelona (Palau Reial)'] == 1) | (imputed_data['station_Barcelona (Poblenou)'] == 1) | (imputed_data['station_Barcelona (Observatori Fabra)'] == 1) | (imputed_data['station_Barcelona (Sants)'] == 1), 1, 0)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_train_test(station, pollutant):
    station_data = imputed_data[imputed_data[station] == 1]

    pollutant_values=station_data[[pollutant, 'Date_time']]
    pollutant_values['Date_time'] = pd.to_datetime(pollutant_values['Date_time'])
    pollutant_values.set_index('Date_time', inplace=True)
    
    # Agregando datos por día
    pollutant_values = pollutant_values.resample('W').mean()

    print(pollutant_values.head())
    #if pollutant_values.index.freq is None:
    #    pollutant_values = pollutant_values.asfreq('D')
    #pollutant_train, pollutant_test = train_test_split(pollutant_values, test_size=0.25, random_state=42)

    split_point = int(len(pollutant_values) * 0.75)
    pollutant_train = pollutant_values[:split_point]
    pollutant_test = pollutant_values[split_point:]

    return pollutant_values, pollutant_train, pollutant_test


station_columns= ['station_Barcelona (Ciutadella)', 'station_Barcelona (Eixample)','station_Barcelona (Gràcia - Sant Gervasi)','station_Barcelona (Observatori Fabra)','station_Barcelona (Palau Reial)','station_Barcelona (Parc Vall Hebron)','station_Barcelona (Poblenou)', 'station_Barcelona (Sants)']
contaminant_columns = ['NO', 'NO2', 'NOX', 'O3', 'PM10']
#contaminant_columns = ['PM10']
type_columns = ['traffic','background']
output_folder = 'arima_plots_weekly'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
results = []

for pollutant in contaminant_columns:
    for station in station_columns:
        
        print(f'Station: {station}')
        print(f'Pollutant: {pollutant}')
        pollutant_values, pollutant_train, pollutant_test = get_train_test(station, pollutant)
        pollutant_train = pollutant_train.asfreq('W')
        
        pollutant_test = pollutant_test.asfreq('W')

        """
        # Asegúrate de que los índices son monotónicos y tienen frecuencia
        if not pollutant_train.index.is_monotonic_increasing:
            pollutant_train = pollutant_train.sort_index()
        if not pollutant_test.index.is_monotonic_increasing:
            pollutant_test = pollutant_test.sort_index()
        
        # Asignar frecuencia diaria si no está presente
        if pollutant_train.index.freq is None:
            pollutant_train.index.freq = pd.infer_freq(pollutant_train.index)
        if pollutant_test.index.freq is None:
            pollutant_test.index.freq = pd.infer_freq(pollutant_test.index)
        """
        """
        plot_acf(pollutant_values)
        plt.title(f'ACF for {pollutant} at {station}')
        plt.savefig(os.path.join(output_folder, f'ACF_{pollutant}_{station}.png'))
        plt.show()
        plt.clf()

        plot_pacf(pollutant_values)
        plt.title(f'PACF for {pollutant} at {station}')
        plt.savefig(os.path.join(output_folder, f'PACF_{pollutant}_{station}.png'))
        plt.show()
        plt.clf()
        """

        """
        # Obtener los 3 inputs del usuario
        input1 = input("Ingrese el primer input (P): ")
        input2 = input("Ingrese el segundo input (D): ")
        input3 = input("Ingrese el tercer input (Q): ")
        
        # Crear una lista con los inputs
        order = (int(input1), int(input2), int(input3))
        """
        stepwise_fit = auto_arima(pollutant_train, seasonal=False, trace=True)
        order = stepwise_fit.order
        print(f'Best ARIMA order: {order} for {pollutant} at {station}')

        model = ARIMA(pollutant_train, order=order)
        model_fit = model.fit()
        print(model_fit.aic)

        model_forecast = model_fit.forecast(steps=pollutant_test.shape[0]) # whats in the [0] is what contains the forecast values

        mae = mean_absolute_error(pollutant_test, model_forecast)
        mse = mean_squared_error(pollutant_test, model_forecast)
        rmse = np.sqrt(mse)
        r2 = r2_score(pollutant_test, model_forecast)
        mape = mean_absolute_percentage_error(pollutant_test, model_forecast)
        bic = model_fit.bic
        aic = model_fit.aic

        plt.figure(figsize=(10, 6))
        plt.scatter(pollutant_test, model_forecast, alpha=0.5)
        plt.plot([pollutant_test.min(), pollutant_test.max()], [pollutant_test.min(), pollutant_test.max()], 'r--')  # Diagonal line for reference
        plt.title(f'Actual vs. Predicted Values for {station}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        plt.savefig(f"arima_plots_weekly/plot_{station}_{pollutant}.png")

        # Guardar los resultados en el DataFrame
        results.append({
            'Station': station,
            'Pollutant': pollutant,
            'Order': order,
            'AIC': aic,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'BIC': bic
        })
        """
        repeat_input = input("¿Quieres repetir la ejecución de las acciones para esta estación y contaminante? (s/n): ").strip().lower()
        if repeat_input != 's':
            repeat = False
        """
        
        
results_df = pd.DataFrame(results)     
results_df.to_csv('auto_arima_results_weekly.csv', index=False)

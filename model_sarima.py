import pandas as pd
import ast
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
from pmdarima import auto_arima
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#################################################################################################
#################################################################################################
## This script performs SARIMAX time series analysis on urban air quality data from Barcelona.
## It uses weekly resampled data from multiple monitoring stations and applies the best ARIMA model parameters to forecast air quality. 
## The script evaluates the forecast accuracy using several metrics, visualizes predictions versus actual values, and stores the results in a CSV file for comprehensive analysis.
#################################################################################################
#################################################################################################


arima_results = pd.read_csv('auto_arima_results.csv')
#arima_results['Order'] = arima_results['Order'].apply(ast.literal_eval)

imputed_data = pd.read_csv('imputed_data.csv')
imputed_data.head()

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_train_test(station, pollutant):
    station_data = imputed_data[imputed_data[station] == 1]

    pollutant_values=station_data[[pollutant, 'Date_time']]
    pollutant_values['Date_time'] = pd.to_datetime(pollutant_values['Date_time'])
    pollutant_values.set_index('Date_time', inplace=True)
    pollutant_values = pollutant_values.resample('W').max()

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
#contaminant_columns = ['NO']
results = []
periodos=[4]


for pollutant in contaminant_columns:
    arima_pollutant_results = arima_results[arima_results['Pollutant'] == pollutant]
    for station in station_columns:
        for x in periodos:
            print(f'Station: {station}')
            print(f'Pollutant: {pollutant}')
            pollutant_values, pollutant_train, pollutant_test = get_train_test(station, pollutant)
            arima_station_results = arima_pollutant_results[arima_pollutant_results['Station'] == station]
            pollutant_train = pollutant_train.asfreq('W')
            pollutant_test = pollutant_test.asfreq('W')
            
            #stepwise_fit = auto_arima(pollutant_train, seasonal=True, m=x, trace=True,                                           max_P=2, max_D=1, max_Q=2, 
            #                             stepwise=True)  # `m` es el periodo de estacionalidad
            #order = stepwise_fit.order
            #seasonal_order = stepwise_fit.seasonal_order

            order = arima_station_results['Order'].values[0].replace("(", "").replace(")", "")
            order=order.split(',')
            seasonal_order = (1,1,1,x)
            
            print(f'Best ARIMA order: {order} and seasonal order: {seasonal_order} for {pollutant} at {station}')
            
            # Ajustar el modelo SARIMAX con los mejores parámetros encontrados
            
            model = SARIMAX(pollutant_train, order= (int(order[1]), int(order[1]), int(order[2])), seasonal_order=seasonal_order)
            model_fit = model.fit(maxiter=35)

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
            plt.savefig(f"plots_sarima_mensual_max/plot_{station}_{pollutant}.png")

            # Guardar los resultados en el DataFrame
            results.append({
                'Station': station,
                'Pollutant': pollutant,
                'Order': order,
                'Seasonal Order': seasonal_order,
                'M':x,
                'AIC': aic,
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape,
                'BIC': bic
            })
            
            
            
# Convertir la lista de resultados en un DataFrame
results_df = pd.DataFrame(results)

# Si el archivo CSV ya existe, leerlo y agregar las nuevas filas
csv_file = 'sarimax_results_mensual_max.csv'
if os.path.exists(csv_file):
    existing_df = pd.read_csv(csv_file)
    updated_df = pd.concat([existing_df, results_df], ignore_index=True)
else:
    updated_df = results_df

# Guardar el DataFrame actualizado en el archivo CSV
updated_df.to_csv(csv_file, index=False)
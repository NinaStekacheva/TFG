import numpy as np
import os
import pandas as pd
import keras_tuner as kt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.regularizers import l2
import wandb
from wandb.keras import WandbCallback
from keras.optimizers import Adam

# Inicializar un nuevo experimento en Weights & Biases
#wandb.init(project='LSTM training' )

#wandbcallback = WandbCallback()
# Cargar datos
imputed_data = pd.read_csv('imputed_data.csv', parse_dates=['Date_time'], index_col='Date_time')

# Crear secuencias para la entrada en LSTM
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps), :])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



def build_model(hp, X_train):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=hp.Int('units', min_value=50, max_value=200, step=50),
                   activation='relu',
                   input_shape=(X_train.shape[1], X_train.shape[2]))))
    model.add(Dense(1))
    model.compile(
        optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss='mean_squared_error')
    return model

station_columns = [
    'station_Barcelona (Ciutadella)', 'station_Barcelona (Eixample)',
    'station_Barcelona (Gràcia - Sant Gervasi)', 'station_Barcelona (Observatori Fabra)',
    'station_Barcelona (Palau Reial)', 'station_Barcelona (Parc Vall Hebron)',
    'station_Barcelona (Poblenou)', 'station_Barcelona (Sants)'
]
contaminant_columns= ['NO', 'NO2', 'NOX' , 'PM10']

results = []


for station in station_columns:
    if station == 'station_Barcelona (Ciutadella)':
         contaminant_columns= ['NO', 'NO2', 'NOX', 'O3']
    elif station == 'station_Barcelona (Sants)':
         contaminant_columns= ['NO', 'NO2', 'NOX']
    elif station == 'station_Barcelona (Poblenou)':
         contaminant_columns= ['NO','NOX' ,'NO2','PM10',]
    else:
        contaminant_columns= ['NO', 'NO2', 'NOX' , 'PM10', 'O3']
    data=imputed_data[imputed_data[station] == 1]
    data = data.resample('W').mean()
    for pollutant in contaminant_columns:
        print(f'Station: {station}')
        print(f'Contaminant: {pollutant}')
        other_pollutants = [p for p in contaminant_columns if p != pollutant]
        # Supongamos que quieres predecir PM10, usaremos NO, NO2, NOX, O3 como features
        data=data[data[station] == 1]
        features = data[other_pollutants]
        target = data[pollutant]

        # Normalizar los datos
        scaler = MinMaxScaler(feature_range=(0, 1))
        features_scaled = scaler.fit_transform(features)
        target_scaled = scaler.fit_transform(target.values.reshape(-1,1))

        time_steps = 3
        X, y = create_dataset(features_scaled, target_scaled, time_steps)

        model = Sequential()
        model.add(Bidirectional(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2]))))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Dividir los datos en entrenamiento y prueba
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Entrenar el modelo
        history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=1, shuffle=False)
        # Hacer predicciones
        y_pred = model.predict(X_test)

        # Desnormalizar los datos si es necesario y evaluar el modelo
        y_test_inv = scaler.inverse_transform(y_test)
        y_pred_inv = scaler.inverse_transform(y_pred)
        # Evaluate forecast
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        r2 = r2_score(y_test_inv, y_pred_inv)
        mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)
        # Número de parámetros
        k = model.count_params()
        # Número de observaciones
        n = X_train.shape[0]
        sse = n * mse
        log_likelihood = -0.5 * n * np.log(2 * np.pi * mse) - (sse / (2 * mse))

        # Calcular BIC
        bic = np.log(n) * k - 2 * log_likelihood
        aic = 2 * k - 2 * log_likelihood
        # Add other metrics and model details as needed

        results.append({
            'Station': station,
            'Pollutant': pollutant,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'AIC': aic,
            'R2': r2,
            'MAPE': mape,
            'BIC': bic
        })

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test_inv, y_pred_inv, alpha=0.5)
        plt.plot([y_test_inv.min(), y_test_inv.max()], [y_test_inv.min(), y_test_inv.max()], 'r--')  # Diagonal line for reference
        plt.title(f'Actual vs. Predicted Values for {station}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        plt.savefig(f"plots_lstm_weekly/plot_{station}_{pollutant}.png")

# Results DataFrame
results_df = pd.DataFrame(results)

# Save results
csv_file = 'lstm_results_weekly.csv'
if os.path.exists(csv_file):
    existing_df = pd.read_csv(csv_file)
    updated_df = pd.concat([existing_df, results_df], ignore_index=True)
else:
    updated_df = results_df
updated_df.to_csv(csv_file, index=False)
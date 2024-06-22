
import pandas as pd

# Read and preprocess data
data = pd.read_csv('imputed_data.csv')
data['Date_time'] = pd.to_datetime(data['Date_time'])
#data.set_index('Date_time', inplace=True)


# Listado de estaciones y contaminantes
stations = [
    'station_Barcelona (Ciutadella)', 'station_Barcelona (Eixample)',
    'station_Barcelona (Gràcia - Sant Gervasi)', 'station_Barcelona (Observatori Fabra)',
    'station_Barcelona (Palau Reial)', 'station_Barcelona (Parc Vall Hebron)',
    'station_Barcelona (Poblenou)', 'station_Barcelona (Sants)'
]
pollutants = ['NO', 'NO2', 'NOX', 'O3', 'PM10']

# Creamos un DataFrame vacío para los resultados
result_df = pd.DataFrame()

# Iteramos sobre cada contaminante
for pollutant in pollutants:
    # Extraemos las columnas relevantes para el contaminante actual
    temp_df = data[['Date_time'] + stations + [pollutant]]
    # Fundimos los datos para llevar las estaciones a filas y transponemos el valor del contaminante
    melted_df = temp_df.melt(id_vars=['Date_time', pollutant], value_vars=stations,
                             var_name='Station', value_name='Station_Value')
    # Filtramos solo las filas donde la estación tiene un 1 (indicando que el dato corresponde a esa estación)
    filtered_df = melted_df[melted_df['Station_Value'] == 1]
    # Renombramos y ajustamos las columnas para el formato final
    filtered_df = filtered_df.rename(columns={pollutant: 'Value'})
    filtered_df['Pollutant'] = pollutant
    # Eliminamos la columna de valor de estación ya que no es necesaria
    filtered_df = filtered_df.drop('Station_Value', axis=1)
    # Agrupamos por fecha y contaminante, pivotando las estaciones a columnas
    pivot_df = filtered_df.pivot_table(index=['Date_time', 'Pollutant'], columns='Station', values='Value', aggfunc='first').reset_index()
    # Añadimos al DataFrame de resultados
    result_df = pd.concat([result_df, pivot_df])

# Reordenamos las columnas para tener 'Date_time' y 'Pollutant' primero
result_df = result_df[['Date_time', 'Pollutant'] + stations]
# Save the result DataFrame to a CSV file
result_df.to_csv('imputed_reordered.csv', index=False)
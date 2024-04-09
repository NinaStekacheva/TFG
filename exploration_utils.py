import pandas as pd



def station_checker(df,station_name):
    #prints na values, and the oldest and earliest date of a station data
    df_station= df[df["NOM ESTACIO"]==station_name]

    num_rows_with_na = df_station.isna().any(axis=1).sum()
    print(f"\nThe station {station_name} has {num_rows_with_na} rows with at least one NA value out of {df_station.shape[0]} rows.")
    df_station["DATA"] = pd.to_datetime(df_station["DATA"])
    print(df_station["DATA"].min())
    

    print(df_station['CONTAMINANT'].unique())
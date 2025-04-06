import numpy as np
import pandas as pd


def create_distances(archivo):
    ## Personalización ##

    carpeta_archivo = "./data/positions/"
    ruta_archivo = carpeta_archivo + archivo + ".txt"

    ruta_guardado_demand = "./data/distances/demand/" + archivo + ".csv"
    ruta_guardado_supply = "./data/distances/supply/" + archivo + ".csv"

    #####################

    # Paso 1: Leer la primera línea para obtener n, m, k
    with open(ruta_archivo, "r") as archivo:
        primera_linea = archivo.readline().strip()
        n, m, k = map(int, primera_linea.split())

    # Leer el archivo completo
    df_sitios_a_proveer = pd.read_csv(
        ruta_archivo, skiprows=1, nrows=n, header=None, sep="\\s+"
    )
    df_sitios_a_proveer.columns = ["x", "y"]

    df_sitios_de_suministro = pd.read_csv(
        ruta_archivo, skiprows=1 + n, nrows=m, header=None, sep="\\s+"
    )
    df_sitios_de_suministro.columns = ["x", "y"]

    proveer_coords = df_sitios_a_proveer[["x", "y"]].to_numpy()
    suministro_coords = df_sitios_de_suministro[["x", "y"]].to_numpy()

    # Calcular la distancia euclidiana entre cada par de puntos (suministro y demanda)
    dist_matrix_demand = np.sqrt(
        ((proveer_coords[:, None, :] - suministro_coords[None, :, :]) ** 2).sum(axis=2)
    )

    dist_df_demand = pd.DataFrame(
        dist_matrix_demand,
        index=df_sitios_a_proveer.index,
        columns=df_sitios_de_suministro.index,
    )

    dist_df_demand.to_csv(ruta_guardado_demand, index=True, header=True)

    # Calcular la distancia euclidiana entre cada par de fábricas
    dist_matrix_supply = np.sqrt(
        ((suministro_coords[:, None, :] - suministro_coords[None, :, :]) ** 2).sum(
            axis=2
        )
    )

    dist_df_supply = pd.DataFrame(
        dist_matrix_supply,
        index=df_sitios_de_suministro.index,
        columns=df_sitios_de_suministro.index,
    )

    dist_df_supply.to_csv(ruta_guardado_supply, index=True, header=True)

    return

import numpy as np
import pandas as pd

## Personalización ##
nombre_archivo = "A3_7500_150_15"
carpeta_archivo = "data/kbcl_instances/two_of/"
ruta_archivo = carpeta_archivo + nombre_archivo + ".txt"

ruta_guardado = "./data/distances/" + nombre_archivo + ".csv"

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

# Calcular la distancia euclidiana entre cada par de puntos
dist_matrix = np.sqrt(
    ((proveer_coords[:, None, :] - suministro_coords[None, :, :]) ** 2).sum(axis=2)
)

# Crear un DataFrame con la matriz de distancias
dist_df = pd.DataFrame(
    dist_matrix, index=df_sitios_a_proveer.index, columns=df_sitios_de_suministro.index
)

# Guardar el dataframe
dist_df.to_csv(ruta_guardado, index=True, header=True)

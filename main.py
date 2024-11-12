import random
import numpy as np
from scipy.spatial import distance
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from tools.utils import genetic_algorithm, plot_results

ruta_archivo = "data/kbcl_instances/two_of/A3_7500_150_15.txt"

# Paso 1: Leer la primera línea para obtener n, m, k
with open(ruta_archivo, "r") as archivo:
    primera_linea = archivo.readline().strip()
    n, m, k = map(int, primera_linea.split())

# Establecer una semilla para asegurar la aleatoriedad reproducible
seed = 42  # Puedes cambiar esta semilla por cualquier número que elijas
np.random.seed(seed)  # Establece la semilla para la aleatoriedad reproducible

# Leer el archivo completo
df_sitios_a_proveer = pd.read_csv(
    ruta_archivo, skiprows=1, nrows=n, header=None, sep="\\s+"
)
df_sitios_a_proveer.columns = ["x", "y"]

df_sitios_de_suministro = pd.read_csv(
    ruta_archivo, skiprows=1 + n, nrows=m, header=None, sep="\\s+"
)
df_sitios_de_suministro.columns = ["x", "y"]

n = 1000
m = 50

df_proveer = df_sitios_a_proveer.sample(n=n, random_state=42)
df_suministro = df_sitios_de_suministro.sample(n=m, random_state=42)

len(df_proveer)

num_generations = 5  # Número de generaciones
population_size = 50  # Tamaño de la población
mutation_rate = 0.1  # Tasa de mutación

best_solution, best_fitness = genetic_algorithm(
    df_suministro,
    df_proveer,
    k,
    num_generations,
    population_size,
    mutation_rate,
)

print("Mejor conjunto de puntos de suministro:", best_solution)
print("Mejor distancia máxima:", best_fitness)

filename = "resultados/soluciones_1.txt"
with open(filename, "a") as file:
    # Añadir una nueva línea con la mejor solución y su fitness
    file.write(f"{best_solution}; {best_fitness}\n")

print(f"Resultados guardados en {filename}")

plot_results(best_solution, df_proveer, df_suministro)

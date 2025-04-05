import numpy as np
import random
import pandas as pd


class Greedy:
    def __init__(self, df_distances_demand, k, m):
        self.df_distances_demand = df_distances_demand
        self.k = k
        self.m = m

    def max_min_dist(self, supply_selected):
        """
        Calcula la máxima de las mínimas distancias de cada punto de demanda a los puntos de suministro seleccionados.
        """
        return self.df_distances_demand.iloc[:, supply_selected].min(axis=1).max()

    def run(self):
        """
        Algoritmo Greedy optimizado para seleccionar puntos de suministro minimizando la peor distancia.
        """
        supply_selected = {random.randrange(0, self.m)}  # Set para búsquedas rápidas
        supply_list = list(supply_selected)  # Lista para mantener orden

        for _ in range(self.k - 1):
            best = np.inf
            best_candidate = None

            # Obtener la distancia mínima actual de cada punto de demanda
            current_min_distances = self.df_distances_demand.iloc[:, supply_list].min(
                axis=1
            )

            for j in range(self.m):
                if j in supply_selected:
                    continue

                # Candidata: obtenemos la mínima distancia considerando este nuevo punto
                new_min_distances = pd.concat(
                    [current_min_distances, self.df_distances_demand.iloc[:, j]], axis=1
                ).min(axis=1)

                value = (
                    new_min_distances.max()
                )  # Aplicamos max_min_dist sin recorrer con for

                if value < best:
                    best = value
                    best_candidate = j

            if best_candidate is not None:
                supply_selected.add(best_candidate)
                supply_list.append(best_candidate)

        return list(supply_selected), best

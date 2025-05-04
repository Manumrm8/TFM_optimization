import random
import pandas as pd


class Greedy:
    def __init__(self, df_distances_demand, k, m, alpha=1.0):
        self.df_distances_demand = df_distances_demand
        self.k = k
        self.m = m
        self.alpha = alpha

    def run(self):
        """
        Algoritmo Greedy con selección aleatoria controlada por alpha y umbral dinámico basado en el valor del mejor candidato.
        """
        supply_selected = {
            random.randrange(0, self.m)
        }  # Para buscar rápido mejor usar set
        supply_list = list(supply_selected)

        for _ in range(self.k - 1):
            current_min_distances = self.df_distances_demand.iloc[:, supply_list].min(
                axis=1
            )
            candidates = []

            for j in range(self.m):
                if j in supply_selected:
                    continue

                new_min_distances = pd.concat(
                    [current_min_distances, self.df_distances_demand.iloc[:, j]], axis=1
                ).min(axis=1)

                value = new_min_distances.max()
                candidates.append((j, value))

            # Obtener mejor y peor valores
            values = [v for _, v in candidates]
            best_value = min(values)
            worst_value = max(values)
            d = worst_value - best_value

            # Umbral de aceptación basado en alpha
            threshold = best_value + (1 - self.alpha) * d

            # Filtrar candidatos que cumplan con el umbral
            eligible_candidates = [j for j, v in candidates if v <= threshold]

            # Seleccionar aleatoriamente uno de los elegibles
            best_candidate = random.choice(eligible_candidates)

            supply_selected.add(best_candidate)
            supply_list.append(best_candidate)

        best = self.df_distances_demand.iloc[:, supply_list].min(axis=1).max()

        return supply_list, best

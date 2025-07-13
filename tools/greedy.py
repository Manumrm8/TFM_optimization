import random
import numpy as np  # Importar numpy


class Greedy:
    def __init__(self, df_distances_demand, k, m, alpha=1.0):
        self.df_distances_demand = df_distances_demand
        self.k = k
        self.m = m
        self.alpha = alpha
        self.num_demand = df_distances_demand.shape[0]

        # Convertir el DataFrame a NumPy array una sola vez para acceso más rápido
        self.distance_matrix = df_distances_demand.values

        # Obtener la lista de todos los índices posibles de supply points
        self.all_supply_indices = list(range(self.m))

    def run(self):
        """
        Algoritmo Greedy optimizado con evaluación incremental usando NumPy.
        Selección aleatoria controlada por alpha y umbral dinámico.
        """
        if self.k <= 0 or self.m <= 0 or self.num_demand <= 0:
            return [], float("inf")  # Casos borde

        # 1. Inicializar con un punto de suministro aleatorio
        # supply_selected = {random.randrange(0, self.m)}
        # supply_list = list(supply_selected)

        # Alternative initial step: Select the facility that minimizes the initial max_min_dist (to itself)
        # which is equivalent to selecting the facility that has the smallest maximum distance to any demand point.
        # Or simply pick the first one for simplicity/speed in the initial step randomness comes later.
        first_selected_index = random.randrange(0, self.m)
        supply_selected = {first_selected_index}
        supply_list = [first_selected_index]

        # Array de NumPy para mantener la distancia mínima actual de cada punto de demanda
        # a los puntos de suministro seleccionados hasta ahora.
        # Inicialmente, es la distancia a la primera facilidad seleccionada.
        current_min_distances_array = self.distance_matrix[
            :, first_selected_index
        ].copy()

        # Usar un conjunto para los índices no seleccionados para búsquedas rápidas
        unselected_indices = set(self.all_supply_indices) - supply_selected

        # 2. Iterativamente seleccionar k-1 puntos adicionales
        # for _ in range(self.k - 1): # Bugfix: should loop until k facilities are selected. If starting with 1, loop k-1 times.
        # Correct loop: continue until k facilities are selected
        while len(supply_list) < self.k:

            # 3. Evaluar candidatos (puntos de suministro no seleccionados)
            candidates = []  # Lista para almacenar (índice_candidato, valor_objetivo)

            # Convertir el conjunto de no seleccionados a lista para iterar si es necesario,
            # pero podemos iterar directamente sobre el conjunto o la lista original 'all_supply_indices'
            # y usar el conjunto 'supply_selected' para el check 'in'.
            # Iterar sobre todos los posibles índices de supply points (0 a m-1)
            for j in self.all_supply_indices:
                if j in supply_selected:
                    continue  # Saltar si ya está seleccionado

                # Calcular el valor objetivo si añadimos el candidato j
                # value_for_j = max(min(current_min_distances_array[i], distance_matrix[i, j]) for i in range(self.num_demand))
                # Esto se hace de forma vectorizada con NumPy:
                potential_min_distances_array = np.minimum(
                    current_min_distances_array, self.distance_matrix[:, j]
                )
                value_for_j = potential_min_distances_array.max()

                candidates.append((j, value_for_j))

            # Si no quedan candidatos (esto no debería pasar si m >= k), salir del bucle
            if not candidates:
                print("Warning: No candidates left before selecting k facilities.")
                break  # Should not happen if m >= k

            # 4. Construir la Lista Restringida de Candidatos (RCL)
            values = [v for _, v in candidates]
            best_value = min(values)
            worst_value = max(values)
            d = worst_value - best_value

            # Umbral de aceptación basado en alpha
            # Si alpha = 1.0, threshold = best_value (Greedy puro)
            # Si alpha = 0.0, threshold = worst_value (Selección aleatoria entre todos los no seleccionados)
            threshold = (
                best_value + self.alpha * d
            )  # Nota: Tu fórmula original era (1 - alpha), ajusté a alpha que es más común para RCL

            # Filtrar candidatos que cumplan con el umbral
            # La RCL contiene candidatos cuya calidad (valor objetivo) está dentro del umbral
            eligible_candidates = [j for j, v in candidates if v <= threshold]

            # Si por alguna razón la RCL está vacía (ej. alpha muy estricto y valores muy juntos),
            # añade el mejor candidato para evitar un error.
            if not eligible_candidates:
                # Encuentra el candidato con el best_value
                best_candidate_fallback = min(candidates, key=lambda item: item[1])[0]
                eligible_candidates.append(best_candidate_fallback)
                # print(f"Warning: RCL vacía con alpha={self.alpha}. Añadiendo el mejor candidato ({best_candidate_fallback}).")

            # 5. Seleccionar aleatoriamente uno de los elegibles de la RCL
            selected_candidate_index = random.choice(eligible_candidates)

            # 6. Añadir el candidato seleccionado a la solución actual
            supply_selected.add(selected_candidate_index)
            supply_list.append(selected_candidate_index)
            unselected_indices.discard(
                selected_candidate_index
            )  # Eliminar del conjunto de no seleccionados

            # 7. **Actualizar incrementalmente las distancias mínimas**
            # La nueva distancia mínima para cada punto de demanda es el mínimo
            # entre su distancia mínima actual y su distancia al nuevo centro seleccionado.
            distances_to_new_facility = self.distance_matrix[
                :, selected_candidate_index
            ]
            current_min_distances_array = np.minimum(
                current_min_distances_array, distances_to_new_facility
            )

        # 8. Calcular el valor objetivo final (max_min_dist)
        final_best_value = current_min_distances_array.max()

        return supply_list, final_best_value

import pandas as pd
from tools.greedy import Greedy
import os
import random
import numpy as np

from tools.local_search import local_search_optimized, local_search_f1_optimized, local_search_f2_optimized, local_search_f3_optimized


####
# Fs#------------------------------------------------------------------------------------------
####


def f1(supply_selected, df_distances_demand):
    """
    - supply_selected: Lista con los indices de los puntos de suministro seleccionados.
    - df_distances_demand: DataFrame con las distancias entre los puntos de suministro y los puntos de demanda.
    """
    value = df_distances_demand.iloc[:, supply_selected].min(axis=1).max()
    return value

def f2(supply_selected, df_distances_demand):
    """
    - supply_selected: Lista con los indices de los puntos de suministro seleccionados.
    - df_distances_demand: DataFrame con las distancias entre los puntos de suministro y los puntos de demanda.
    """
    asignacion = df_distances_demand.iloc[:, supply_selected].idxmin(axis=1)
    maximum = asignacion.value_counts().max()
    return maximum

def f3(supply_selected, df_distances_demand):
    """
    - supply_selected: Lista con los indices de los puntos de suministro seleccionados.
    - df_distances_demand: DataFrame con las distancias entre los puntos de suministro y los puntos de demanda.
    """
    asignacion = df_distances_demand.iloc[:, supply_selected].idxmin(axis=1)
    counts = asignacion.value_counts()
    return counts.max() - counts.min()

####
# Añadir soluciones#------------------------------------------------------------------------------------------
####

def add_solutions_optimized(supply_selected, f1_value, f2_value, f3_value, route_solutions, df_solutions):
    """
    - supply_selected: Lista con los indices de los puntos de suministro seleccionados.
    - f1_value: Valor de la función objetivo f1 del supply_selected.
    - f2_value: Valor de la función objetivo f2 del supply_selected.
    - f3_value: Valor de la función objetivo f3 del supply_selected.
    - route_solutions: Ruta donde se encuentra el archivo csv de las soluciones.
    - df_solutions: DataFrame que contiene las soluciones del frente de pareto actual.
    """
    supply_selected = str(sorted(supply_selected))

    # --- CASO BASE: Si el DataFrame está vacío, añadir y salir ---
    if df_solutions.empty:
        df_solutions = pd.DataFrame(
            [{"solution": supply_selected, "f1": f1_value, "f2": f2_value, "f3": f3_value}]
        )
        df_solutions.to_csv(route_solutions, index=False)
        return True

    # Usar un set para comprobar existencia (O(1) en promedio) ---
    existing_solutions = set(df_solutions["solution"])
    if supply_selected in existing_solutions:
        return False # La solución ya existe


    # Extraer los valores numéricos a un array de NumPy para cálculos rápidos
    vals = df_solutions[["f1", "f2", "f3"]].values
    new_vals = np.array([f1_value, f2_value, f3_value])


    if np.any(np.all(vals <= new_vals, axis=1)):
        return False # La nueva solución es dominada, no hacer nada

    # 2. Si no es dominada, eliminar las soluciones que SÍ son dominadas por la nueva.
    #    Una solución 'new' domina a una existente 's' si:
    #    (new_f1 <= s_f1, new_f2 <= s_f2, new_f3 <= s_f3) Y
    #    (new_f1 < s_f1  O  new_f2 < s_f2  O  new_f3 < s_f3)
    #    Lo calculamos con una máscara booleana.
    
    # Condición 1: Todos los valores de la nueva solución son menores o iguales
    # a los de las soluciones existentes.
    cond1 = np.all(new_vals <= vals, axis=1)
    # Condición 2: Al menos un valor de la nueva solución es estrictamente menor.
    cond2 = np.any(new_vals < vals, axis=1)
    
    # La máscara final identifica las filas a eliminar (las dominadas)
    dominated_mask = cond1 & cond2
    
    # Mantener solo las filas que NO están dominadas
    df_solutions = df_solutions[~dominated_mask]

    # 3. Añadir la nueva solución no dominada
    new_solution_df = pd.DataFrame(
        [{"solution": supply_selected, "f1": f1_value, "f2": f2_value, "f3": f3_value}]
    )
    df_solutions = pd.concat([df_solutions, new_solution_df], ignore_index=True)
    
    # Guardar el resultado final en el CSV
    df_solutions.to_csv(route_solutions, index=False)
    
    return True


####
# Multi Armed Bandit#------------------------------------------------------------------------------------------
####


def select_arm(context, betha, n_arms, weights, temperature=1.0):
    if np.random.rand() < betha:  # Exploración: acción aleatoria
        return np.random.randint(0, n_arms)
    else:  # Explotación: probabilidad de realizar la acción según el modelo
        context_np = np.array(context).reshape(1, -1)  # Convertir a fila vector

        context_np = context_np / np.sum(context_np)  # Normalización del contexto

        expected_rewards = np.dot(context_np, weights).flatten()

        # Aplicar softmax
        exp_rewards = np.exp(
            (expected_rewards - np.max(expected_rewards)) / temperature
        )  # Ajustar temperatura para mejorar exploración o explotación

        probabilities = exp_rewards / np.sum(exp_rewards)

        if np.isnan(probabilities).all() or np.sum(probabilities) == 0:
            return np.random.choice(n_arms)
        try:
            choice = np.random.choice(n_arms, p=probabilities)
        except Exception as e:
            choice = np.random.choice(n_arms)

        # Seleccionar un brazo aleatoriamente basado en estas probabilidades
        return choice


def decode_action(chosen_arm, parameters=["f1", "f2", "f3"], k_values=[1, 2, 3, 4, 5]):
    param_idx = chosen_arm // len(k_values)
    k_idx = chosen_arm % len(k_values)
    return parameters[param_idx], k_values[k_idx]


def contexto(f1, f2, f3, df_solutions):
    """
    Calcula la mínima reducción necesaria en f1, f2 o f3 para que la
    combinación no sea dominada por ninguna solución existente en df_solutions.

    Args:
        f1 (float): Valor del primer objetivo.
        f2 (float): Valor del segundo objetivo.
        f3 (float): Valor del tercer objetivo.
        df_solutions (pd.DataFrame): DataFrame con las soluciones existentes.

    Returns:
        tuple: Una tupla (d1, d2, d3) con la reducción  necesaria en cada dimensión.
    """
    # 1. Identificar las soluciones que dominan la combinación actual.
    # Una solución 's' domina a la actual 'c' si s_f1 <= c_f1, s_f2 <= c_f2, Y s_f3 <= c_f3.
    dominating_solutions = df_solutions[
        (df_solutions["f1"] <= f1)
        & (df_solutions["f2"] <= f2)
        & (df_solutions["f3"] <= f3)
    ]

    # Aquí entra cuando es una solución existente
    if dominating_solutions.empty:
        return (0, 0, 0), 1

    # 3. Si hay soluciones dominantes, calcular las diferencias.
    # Estas son las "distancias" que necesitamos superar en cada dimensión.
    delta_f1 = f1 - dominating_solutions["f1"]
    delta_f2 = f2 - dominating_solutions["f2"]
    delta_f3 = f3 - dominating_solutions["f3"]

    # 4. Encontrar la mínima diferencia GLOBAL.
    # Esto representa la reducción más "barata" que podemos hacer para
    # que nuestra combinación deje de ser dominada por al menos una de las soluciones.
    min_delta_f1 = delta_f1.min()
    min_delta_f2 = delta_f2.min()
    min_delta_f3 = delta_f3.min()

    return (min_delta_f1, min_delta_f2, min_delta_f3)


def update_weights(weights, chosen_arm, reward, context, learning_rate):
    """
    Actualiza los pesos del brazo elegido basándose en la recompensa y el contexto.
    
    Args:
        weights (np.array): Matriz de pesos (3x15).
        chosen_arm (int): Índice del brazo seleccionado.
        reward (int): Recompensa recibida (+5 o -1).
        context (list or np.array): Vector de contexto de 3 elementos.
        learning_rate (float): Tasa de aprendizaje.
    """
    context_np = np.array(context)
    
    # La actualización clave: simple, sin normalización.
    # Los pesos de la columna (brazo) elegida se actualizan.
    # Esto permite que los pesos crezcan o decrezcan sin límite, acumulando el aprendizaje.
    weights[:, chosen_arm] += learning_rate * reward * context_np
    
    return weights


####
# GRASP#------------------------------------------------------------------------------------------
####


def multi_GRASP_Bandit(
    archive,
    k,
    m,
    context_size,
    max_iterations=5,
    alpha=1.0,
    betha=0.2,
    learning_rate=1,
    i=0,
):

    n_arms = context_size * max_iterations

    folder_distances = "./data/distances/demand/"
    route_distances = folder_distances + archive + ".csv"
    df_distances_demand = pd.read_csv(route_distances, index_col=0)

    folder_solutions = f"Solutions/Multiprocessing/{archive}/"
    # Crea la carpeta si no existe
    os.makedirs(folder_solutions, exist_ok=True)

    route_solutions = folder_solutions + archive + f"_#{i}" + ".csv"

    w_dir = f"Weights/{archive}/"
    os.makedirs(w_dir, exist_ok=True)
    route_weights = w_dir + archive + f"_#{i}" + ".npy"

    if os.path.exists(route_solutions):
        df_solutions = pd.read_csv(route_solutions)
    else:
        columnas = ["solution", "f1", "f2", "f3"]
        df_solutions = pd.DataFrame(columns=columnas)

    if os.path.exists(route_weights):
        weights = np.load(route_weights)
    else:
        weights = np.zeros((n_arms, context_size))
        print(f"No había pesos")

    greedy_algorithm = Greedy(df_distances_demand, k, m, alpha)
    """
    Algoritmo GRASP con dos búsquedas locales elegidas aleatoriamente (sin repetición).
    Se mide y muestra el tiempo de ejecución de cada búsqueda local.
    """
    # Etapa Greedy inicial
    solution, f1_value = greedy_algorithm.run()

    solution=local_search_optimized(solution, df_distances_demand, k, m)

    f1_value = f1(solution, df_distances_demand)
    f2_value = f2(solution, df_distances_demand)
    f3_value = f3(solution, df_distances_demand)

    solucion_encontrada = add_solutions_optimized(
        solution, f1_value, f2_value, f3_value, route_solutions, df_solutions
    )

    if solucion_encontrada:
        print(f"hay nueva solucion: {solution}")
        return solution
    else:
        context = contexto(
            f1_value, f2_value, f3_value, df_solutions
        )

    # Empiezo bucle
    reward = 1
    count = 0
    acciones_escogidas = []
    while (
        reward > 0 and count <= m * k
    ):  # Limito a que la recompensa deje de mejorar, o haga m*k iteraciones
        # Necesito el context nuevo, pesos nuevos,

        chosen_arm = select_arm(context, betha, n_arms, weights)
        funcion, n_veces = decode_action(
            chosen_arm, parameters=["f1", "f2", "f3"], k_values=[1, 2, 3, 4, 5]
        )
        acciones_escogidas.append(f"mejorar{funcion} {n_veces} veces")

        # Ejecutar primera búsqueda local
        if funcion == "f1":
            solution, f1_value = local_search_f1_optimized(
                solution, df_distances_demand, k, m, n_veces
            )

        elif funcion == "f2":
            solution, f2_value = local_search_f2_optimized(
                solution, df_distances_demand, k, m, n_veces
            )
        
        elif funcion == "f3":
            solution, f3_value = local_search_f3_optimized(
                solution, df_distances_demand, k, m, n_veces
            )
            
        solution=local_search_optimized(solution, df_distances_demand, k, m)
        
        f1_value = f1(solution, df_distances_demand)
        f2_value = f2(solution, df_distances_demand)
        f3_value = f3(solution, df_distances_demand)

        solucion_encontrada = add_solutions_optimized(
            solution, f1_value, f2_value, f3_value, route_solutions, df_solutions
        )

        if solucion_encontrada:
            print(f"hay nueva solucion: {solution}")
            reward = 5
            weights = update(
                chosen_arm, context, reward, weights, route_weights, learning_rate
            )
            break
        else:
            context = contexto(
                f1_value, f2_value, f3_value, df_solutions
            )
            reward = -1

        weights = update(
            chosen_arm, context, reward, weights, route_weights, learning_rate
        )
        count += 1

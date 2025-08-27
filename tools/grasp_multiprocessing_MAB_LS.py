import pandas as pd
from tools.greedy import Greedy
import os
import random
import numpy as np

from tools.local_search import local_search_optimized, local_search_f1_optimized, local_search_f2_optimized, local_search_f3_optimized


####
# F#------------------------------------------------------------------------------------------
####


def f1(supply_selected, df_distances_demand):
    """
    - df_distances_demand: DataFrame con las distancias entre los puntos de suministro y los puntos de demanda.
    - supply_selected: Lista con los indices de los puntos de suministro seleccionados.
    """

    value = df_distances_demand.iloc[:, supply_selected].min(axis=1).max()
    return value

def f2(supply_selected, df_distances_demand):
    """
    - df_distances_demand: DataFrame con las distancias entre los puntos de suministro y los puntos de demanda.
    - supply_selected: Lista con los indices de los puntos de suministro seleccionados.
    """
    asignacion = df_distances_demand.iloc[:, supply_selected].idxmin(axis=1)
    maximum = asignacion.value_counts().max()
    return maximum

def f3(supply_selected, df_distances_demand):
    """
    Calcula la diferencia entre el número máximo y mínimo de demandas asignadas a los puntos de suministro seleccionados,
    de forma más eficiente.
    """
    asignacion = df_distances_demand.iloc[:, supply_selected].idxmin(axis=1)
    counts = asignacion.value_counts()
    return counts.max() - counts.min()

####
# Añadir soluciones#------------------------------------------------------------------------------------------
####

def add_solutions(solution, f1, f2, f3, route_solutions, df_solutions):
    solution = str(sorted(solution))
    solucion_encontrada = False

    if not df_solutions.empty:
        if solution in df_solutions["solution"].values:
            return  # La solución ya existe, no hacer nada

        df_dominado = df_solutions.copy()
        df_dominado = df_dominado[df_dominado["f1"] <= f1]
        df_dominado = df_dominado[df_dominado["f2"] <= f2]
        df_dominado = df_dominado[df_dominado["f3"] <= f3]
        if df_dominado.empty:
            # Eliminar soluciones que sean dominadas por la nueva
            df_solutions = df_solutions[
                ~(
                    (df_solutions["f1"] >= f1)
                    & (df_solutions["f2"] >= f2)
                    & (df_solutions["f3"] >= f3)
                    & (
                        (df_solutions["f1"] > f1)
                        | (df_solutions["f2"] > f2)
                        | (df_solutions["f3"] > f3)
                    )
                )
            ]
            # Agregar la nueva solución
            new_solution = pd.DataFrame(
                [{"solution": solution, "f1": f1, "f2": f2, "f3": f3}]
            )
            df_solutions = pd.concat([df_solutions, new_solution], ignore_index=True)
            df_solutions.to_csv(route_solutions, index=False)
            solucion_encontrada = True
    else:
        df_solutions = pd.DataFrame(
            [{"solution": solution, "f1": f1, "f2": f2, "f3": f3}]
        )
        df_solutions.to_csv(route_solutions, index=False)
    return solucion_encontrada


####
# Multi Armed Bandit#------------------------------------------------------------------------------------------
####


def select_arm(context, betha, n_arms, weights, temperature=1.0):
    if np.random.rand() < betha:  # Exploración: acción aleatoria
        return np.random.randint(0, n_arms)
    else:  # Explotación: probabilidad de realizar la acción según el modelo
        context_np = np.array(context).reshape(1, -1)  # Convertir a fila vector

        context_np = context_np / np.sum(context_np)  # Normalización del contexto

        expected_rewards = np.dot(weights, context_np.T).flatten()

        # Aplicar softmax
        exp_rewards = np.exp(
            (expected_rewards - np.max(expected_rewards)) / temperature
        )  # Ajustar temperatura para mejorar exploración o explotación

        sum_exp_rewards = np.sum(exp_rewards)
        if sum_exp_rewards == 0:
            choice = np.random.choice(n_arms)
        else:
            try:
                probabilities = exp_rewards / np.sum(exp_rewards)
                choice = np.random.choice(n_arms, p=probabilities)
            except Exception as e:
                print(e)
                print(weights)
                print(probabilities)
                choice = np.random.choice(n_arms)

            # Seleccionar un brazo aleatoriamente basado en estas probabilidades
            return choice


def decode_action(chosen_arm, parameters=["f1", "f2", "f3"], k_values=[1, 2, 3, 4, 5]):
    param_idx = chosen_arm // len(k_values)
    k_idx = chosen_arm % len(k_values)
    return parameters[param_idx], k_values[k_idx]


def contexto_and_evaluate(f1, f2, f3, df_solutions):
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

    v1 = f1 / (f1 + min_delta_f1)
    v2 = f2 / (f2 + min_delta_f2)
    v3 = f3 / (f3 + min_delta_f3)
    value = (v1 + v2 + v3) / 3

    return (min_delta_f1, min_delta_f2, min_delta_f3), value


def update(chosen_arm, context, reward, weights, route_weights, learning_rate):
    # Asegurarse de que el contexto sea un array numpy
    context_np = np.array(context)

    weights[chosen_arm] += learning_rate * reward * context_np

    min_val = np.min(weights)
    max_val = np.max(weights)

    # 2. Manejar el caso donde todos los valores son idénticos para evitar división por cero
    if max_val == min_val:
        return np.full(weights.shape, 0)

    # 3. Aplicar la fórmula de normalización Min-Max
    reescaled_weights = (weights - min_val) / (max_val - min_val)
    reescaled_weights = np.round(reescaled_weights, 2)

    np.save(route_weights, reescaled_weights)

    return reescaled_weights


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

    solucion_encontrada = add_solutions(
        solution, f1_value, f2_value, f3_value, route_solutions, df_solutions
    )

    if solucion_encontrada:
        print(f"hay nueva solucion: {solution}")
        return solution
    else:
        context, value_prev = contexto_and_evaluate(
            f1_value, f2_value, f3_value, df_solutions
        )

    # Empiezo bucle
    reward = 1
    count = 0
    values = [value_prev]
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

        solucion_encontrada = add_solutions(
            solution, f1_value, f2_value, f3_value, route_solutions, df_solutions
        )

        if solucion_encontrada:
            print(f"hay nueva solucion: {solution}")
            reward = 1
            weights = update(
                chosen_arm, context, reward, weights, route_weights, learning_rate
            )
            break
        else:
            new_context, value_next = contexto_and_evaluate(
                f1_value, f2_value, f3_value, df_solutions
            )
            reward = (value_next - value_prev) / 3

        weights = update(
            chosen_arm, context, reward, weights, route_weights, learning_rate
        )

        context = new_context
        value_prev = value_next
        values.append(value_prev)
        count += 1

import pandas as pd
from tools.greedy_opt import Greedy
import os
import random


####
# F1#------------------------------------------------------------------------------------------
####


def f1(supply_selected, df_distances_demand):
    """
    - df_distances_demand: DataFrame con las distancias entre los puntos de suministro y los puntos de demanda.
    - supply_selected: Lista con los indices de los puntos de suministro seleccionados.
    """

    value = df_distances_demand.iloc[:, supply_selected].min(axis=1).max()
    return value


def max_min_dist(supply_selected, df_distances_demand):
    """
    Calcula la máxima de las mínimas distancias de cada punto de demanda a los puntos de suministro seleccionados.
    """
    return df_distances_demand.iloc[:, supply_selected].min(axis=1).max()


def local_search_f1(solution, value, m, df_distances_demand):
    """
    Realiza una búsqueda local para mejorar la solución utilizando el enfoque optimizado de max_min_dist.
    """
    best_solution = solution[:]
    best_objective = value

    for i in range(len(solution)):
        for j in range(m):
            if j not in solution:
                temp_solution = solution[:]
                temp_solution[i] = j
                temp_objective = max_min_dist(temp_solution, df_distances_demand)

                if temp_objective < best_objective:
                    best_objective = temp_objective
                    best_solution = temp_solution[:]

    return best_solution, best_objective


####
# F2#------------------------------------------------------------------------------------------
####


def f2(supply_selected, df_distances_demand):
    """
    - df_distances_demand: DataFrame con las distancias entre los puntos de suministro y los puntos de demanda.
    - supply_selected: Lista con los indices de los puntos de suministro seleccionados.
    """
    asignacion = df_distances_demand.iloc[:, supply_selected].idxmin(axis=1)
    maximum = asignacion.value_counts().max()
    return maximum


def max_demand_per_supply(supply_selected, df_distances_demand):
    """
    Calcula el número máximo de demandas asignadas a un único punto de suministro.
    """
    asignacion = df_distances_demand.iloc[:, supply_selected].idxmin(axis=1)
    return asignacion.value_counts().max()


def local_search_f2(solution, value, m, df_distances_demand):
    """
    Realiza una búsqueda local para mejorar la solución minimizando el máximo número de demandas
    asignadas a un único punto de suministro.
    """
    best_solution = solution[:]
    best_objective = value

    for i in range(len(solution)):
        for j in range(m):
            if j not in solution:
                temp_solution = solution[:]
                temp_solution[i] = j
                temp_objective = max_demand_per_supply(
                    temp_solution, df_distances_demand
                )

                if temp_objective < best_objective:
                    best_objective = temp_objective
                    best_solution = temp_solution[:]

    return best_solution, best_objective


####
# F3#------------------------------------------------------------------------------------------
####


def f3(supply_selected, df_distances_demand):
    """
    Calcula la diferencia entre el número máximo y mínimo de demandas asignadas a los puntos de suministro seleccionados,
    de forma más eficiente.
    """
    asignacion = df_distances_demand.iloc[:, supply_selected].idxmin(axis=1)
    counts = asignacion.value_counts()
    return counts.max() - counts.min()


def local_search_f3(solution, value, m, df_distances_demand):
    """
    Realiza una búsqueda local para mejorar la solución minimizando la diferencia entre el número
    máximo y mínimo de demandas asignadas a un punto de suministro.
    """
    best_solution = solution[:]
    best_objective = value

    for i in range(len(solution)):
        for j in range(m):
            if j not in solution:
                temp_solution = solution[:]
                temp_solution[i] = j
                temp_objective = f3(temp_solution, df_distances_demand)

                if temp_objective < best_objective:
                    best_objective = temp_objective
                    best_solution = temp_solution[:]

    return best_solution, best_objective


def add_solutions(solution, f1, f2, f3, route_solutions, df_solutions):
    solution = str(sorted(solution))

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
            print(new_solution)
            df_solutions = pd.concat([df_solutions, new_solution], ignore_index=True)
            df_solutions.to_csv(route_solutions, index=False)
    else:
        df_solutions = pd.DataFrame(
            [{"solution": solution, "f1": f1, "f2": f2, "f3": f3}]
        )
        df_solutions.to_csv(route_solutions, index=False)
    return


def multi_GRASP(archive, k, m, alpha=1.0):

    folder_distances = "./data/distances/demand/"
    route_distances = folder_distances + archive + ".csv"
    df_distances_demand = pd.read_csv(route_distances)

    folder_solutions = "Solutions/"
    route_solutions = folder_solutions + archive + ".csv"
    if os.path.exists(route_solutions):
        df_solutions = pd.read_csv(route_solutions)
        print(df_solutions)
    else:
        columnas = ["solution", "f1", "f2", "f3"]
        df_solutions = pd.DataFrame(columns=columnas)
        print(df_solutions)

    k = k
    m = m
    greedy_algorithm = Greedy(df_distances_demand, k, m, alpha)
    """
    Algoritmo GRASP con dos búsquedas locales elegidas aleatoriamente (sin repetición).
    Se mide y muestra el tiempo de ejecución de cada búsqueda local.
    """
    # Etapa Greedy inicial
    solution, f1_value = greedy_algorithm.run()
    f2_value = f2(solution, df_distances_demand)
    f3_value = f3(solution, df_distances_demand)

    add_solutions(solution, f1_value, f2_value, f3_value, route_solutions, df_solutions)

    # Lista de búsquedas locales disponibles
    local_searches = [
        ("f1", local_search_f1),
        ("f2", local_search_f2),
        ("f3", local_search_f3),
    ]

    # Elegir dos búsquedas diferentes aleatoriamente
    first_name, first_ls = random.choice(local_searches)
    remaining_searches = [ls for ls in local_searches if ls[0] != first_name]
    second_name, second_ls = random.choice(remaining_searches)

    # Ejecutar primera búsqueda local
    if first_name == "f1":
        solution, f1_value = first_ls(solution, f1_value, m, df_distances_demand)
        f2_value = f2(solution, df_distances_demand)
        f3_value = f3(solution, df_distances_demand)
    elif first_name == "f2":
        solution, f2_value = first_ls(solution, f2_value, m, df_distances_demand)
        f1_value = f1(solution, df_distances_demand)
        f3_value = f3(solution, df_distances_demand)
    elif first_name == "f3":
        solution, f3_value = first_ls(solution, f3_value, m, df_distances_demand)
        f1_value = f1(solution, df_distances_demand)
        f2_value = f2(solution, df_distances_demand)

    add_solutions(solution, f1_value, f2_value, f3_value, route_solutions, df_solutions)

    if second_name == "f1":
        solution, f1_value = second_ls(solution, f1_value, m, df_distances_demand)
        f2_value = f2(solution, df_distances_demand)
        f3_value = f3(solution, df_distances_demand)
    elif second_name == "f2":
        solution, f2_value = second_ls(solution, f2_value, m, df_distances_demand)
        f1_value = f1(solution, df_distances_demand)
        f3_value = f3(solution, df_distances_demand)
    elif second_name == "f3":
        solution, f3_value = second_ls(solution, f3_value, m, df_distances_demand)
        f1_value = f1(solution, df_distances_demand)
        f2_value = f2(solution, df_distances_demand)

    add_solutions(solution, f1_value, f2_value, f3_value, route_solutions, df_solutions)

    return solution

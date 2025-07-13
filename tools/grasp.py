import pandas as pd
from tools.greedy import Greedy
import os
import random


class Grasp:
    def __init__(self, archive, k, m, alpha=1.0):
        folder_distances = "./data/distances/demand/"
        route_distances = folder_distances + archive + ".csv"
        self.df_distances_demand = pd.read_csv(route_distances)

        folder_solutions = "Solutions/"
        self.route_solutions = folder_solutions + archive + ".csv"
        if os.path.exists(self.route_solutions):
            self.df_solutions = pd.read_csv(self.route_solutions)
            print(self.df_solutions)
        else:
            columnas = ["solution", "f1", "f2", "f3"]
            self.df_solutions = pd.DataFrame(columns=columnas)
            print(self.df_solutions)

        self.k = k
        self.m = m
        self.greedy_algorithm = Greedy(self.df_distances_demand, k, m, alpha)

    ####
    # F1#------------------------------------------------------------------------------------------
    ####

    def f1(self, supply_selected):
        """
        - df_distances_demand: DataFrame con las distancias entre los puntos de suministro y los puntos de demanda.
        - supply_selected: Lista con los indices de los puntos de suministro seleccionados.
        """

        value = self.df_distances_demand.iloc[:, supply_selected].min(axis=1).max()
        return value

    def max_min_dist(self, supply_selected):
        """
        Calcula la máxima de las mínimas distancias de cada punto de demanda a los puntos de suministro seleccionados.
        """
        return self.df_distances_demand.iloc[:, supply_selected].min(axis=1).max()

    def local_search_f1(self, solution, value):
        """
        Realiza una búsqueda local para mejorar la solución utilizando el enfoque optimizado de max_min_dist.
        """
        best_solution = solution[:]
        best_objective = value

        for i in range(len(solution)):
            for j in range(self.m):
                if j not in solution:
                    temp_solution = solution[:]
                    temp_solution[i] = j
                    temp_objective = self.max_min_dist(temp_solution)

                    if temp_objective < best_objective:
                        best_objective = temp_objective
                        best_solution = temp_solution[:]

        return best_solution, best_objective

    ####
    # F2#------------------------------------------------------------------------------------------
    ####

    def f2(self, supply_selected):
        """
        - df_distances_demand: DataFrame con las distancias entre los puntos de suministro y los puntos de demanda.
        - supply_selected: Lista con los indices de los puntos de suministro seleccionados.
        """
        asignacion = self.df_distances_demand.iloc[:, supply_selected].idxmin(axis=1)
        maximum = asignacion.value_counts().max()
        return maximum

    def max_demand_per_supply(self, supply_selected):
        """
        Calcula el número máximo de demandas asignadas a un único punto de suministro.
        """
        asignacion = self.df_distances_demand.iloc[:, supply_selected].idxmin(axis=1)
        return asignacion.value_counts().max()

    def local_search_f2(self, solution, value):
        """
        Realiza una búsqueda local para mejorar la solución minimizando el máximo número de demandas
        asignadas a un único punto de suministro.
        """
        best_solution = solution[:]
        best_objective = value

        for i in range(len(solution)):
            for j in range(self.m):
                if j not in solution:
                    temp_solution = solution[:]
                    temp_solution[i] = j
                    temp_objective = self.max_demand_per_supply(temp_solution)

                    if temp_objective < best_objective:
                        best_objective = temp_objective
                        best_solution = temp_solution[:]

        return best_solution, best_objective

    ####
    # F3#------------------------------------------------------------------------------------------
    ####

    def f3(self, supply_selected):
        """
        Calcula la diferencia entre el número máximo y mínimo de demandas asignadas a los puntos de suministro seleccionados,
        de forma más eficiente.
        """
        asignacion = self.df_distances_demand.iloc[:, supply_selected].idxmin(axis=1)
        counts = asignacion.value_counts()
        return counts.max() - counts.min()

    def local_search_f3(self, solution, value):
        """
        Realiza una búsqueda local para mejorar la solución minimizando la diferencia entre el número
        máximo y mínimo de demandas asignadas a un punto de suministro.
        """
        best_solution = solution[:]
        best_objective = value

        for i in range(len(solution)):
            for j in range(self.m):
                if j not in solution:
                    temp_solution = solution[:]
                    temp_solution[i] = j
                    temp_objective = self.f3(temp_solution)

                    if temp_objective < best_objective:
                        best_objective = temp_objective
                        best_solution = temp_solution[:]

        return best_solution, best_objective

    def add_solutions(self, solution, f1, f2, f3):
        solution = str(sorted(solution))

        if not self.df_solutions.empty:
            if solution in self.df_solutions["solution"].values:
                return  # La solución ya existe, no hacer nada

            df_dominado = self.df_solutions.copy()
            df_dominado = df_dominado[df_dominado["f1"] <= f1]
            df_dominado = df_dominado[df_dominado["f2"] <= f2]
            df_dominado = df_dominado[df_dominado["f3"] <= f3]
            if df_dominado.empty:
                # Eliminar soluciones que sean dominadas por la nueva
                self.df_solutions = self.df_solutions[
                    ~(
                        (self.df_solutions["f1"] >= f1)
                        & (self.df_solutions["f2"] >= f2)
                        & (self.df_solutions["f3"] >= f3)
                        & (
                            (self.df_solutions["f1"] > f1)
                            | (self.df_solutions["f2"] > f2)
                            | (self.df_solutions["f3"] > f3)
                        )
                    )
                ]
                # Agregar la nueva solución
                new_solution = pd.DataFrame(
                    [{"solution": solution, "f1": f1, "f2": f2, "f3": f3}]
                )
                print(new_solution)
                self.df_solutions = pd.concat(
                    [self.df_solutions, new_solution], ignore_index=True
                )
                self.df_solutions.to_csv(self.route_solutions, index=False)
        else:
            self.df_solutions = pd.DataFrame(
                [{"solution": solution, "f1": f1, "f2": f2, "f3": f3}]
            )
            self.df_solutions.to_csv(self.route_solutions, index=False)
        return

    def run(self):
        """
        Algoritmo GRASP con dos búsquedas locales elegidas aleatoriamente (sin repetición).
        Se mide y muestra el tiempo de ejecución de cada búsqueda local.
        """
        # Etapa Greedy inicial
        solution, f1_value = self.greedy_algorithm.run()
        f2_value = self.f2(solution)
        f3_value = self.f3(solution)

        self.add_solutions(solution, f1_value, f2_value, f3_value)

        # Lista de búsquedas locales disponibles
        local_searches = [
            ("f1", self.local_search_f1),
            ("f2", self.local_search_f2),
            ("f3", self.local_search_f3),
        ]

        # Elegir dos búsquedas diferentes aleatoriamente
        first_name, first_ls = random.choice(local_searches)
        remaining_searches = [ls for ls in local_searches if ls[0] != first_name]
        second_name, second_ls = random.choice(remaining_searches)

        # Ejecutar primera búsqueda local
        if first_name == "f1":
            solution, f1_value = first_ls(solution, f1_value)
            f2_value = self.f2(solution)
            f3_value = self.f3(solution)
        elif first_name == "f2":
            solution, f2_value = first_ls(solution, f2_value)
            f1_value = self.f1(solution)
            f3_value = self.f3(solution)
        elif first_name == "f3":
            solution, f3_value = first_ls(solution, f3_value)
            f1_value = self.f1(solution)
            f2_value = self.f2(solution)

        self.add_solutions(solution, f1_value, f2_value, f3_value)

        if second_name == "f1":
            solution, f1_value = second_ls(solution, f1_value)
            f2_value = self.f2(solution)
            f3_value = self.f3(solution)
        elif second_name == "f2":
            solution, f2_value = second_ls(solution, f2_value)
            f1_value = self.f1(solution)
            f3_value = self.f3(solution)
        elif second_name == "f3":
            solution, f3_value = second_ls(solution, f3_value)
            f1_value = self.f1(solution)
            f2_value = self.f2(solution)

        self.add_solutions(solution, f1_value, f2_value, f3_value)

        return

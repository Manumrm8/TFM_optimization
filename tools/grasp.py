import pandas as pd
from tools.greedy import Greedy
import os


class Grasp:
    def __init__(self, archive, k, m):
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
        self.greedy_algorithm = Greedy(self.df_distances_demand, k, m)

    def f1(self, supply_selected):
        """
        - df_distances_demand: DataFrame con las distancias entre los puntos de suministro y los puntos de demanda.
        - supply_selected: Lista con los indices de los puntos de suministro seleccionados.
        """

        dist = []
        for demand_point in self.df_distances_demand.index:
            distancia = self.df_distances_demand.iloc[
                demand_point, supply_selected
            ].min()
            dist.append(distancia)
        return max(dist)

    def f2(self, supply_selected):
        """
        - df_distances_demand: DataFrame con las distancias entre los puntos de suministro y los puntos de demanda.
        - supply_selected: Lista con los indices de los puntos de suministro seleccionados.
        """
        asignacion = self.df_distances_demand.iloc[:, supply_selected].idxmin(axis=1)
        maximum = asignacion.value_counts().max()
        return maximum

    def f3(self, supply_selected):
        """
        - df_distances_demand: DataFrame con las distancias entre los puntos de suministro y los puntos de demanda.
        - supply_selected: Lista con los indices de los puntos de suministro seleccionados.
        """
        asignacion = self.df_distances_demand.iloc[:, supply_selected].idxmin(axis=1)
        maximum = asignacion.value_counts().max()
        minimum = asignacion.value_counts().min()
        return maximum - minimum

    def max_min_dist(self, supply_selected):
        """
        Calcula la máxima de las mínimas distancias de cada punto de demanda a los puntos de suministro seleccionados.
        """
        return self.df_distances_demand.iloc[:, supply_selected].min(axis=1).max()

    def local_search(self, solution, value):
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
        Algoritmo GRASP
        """
        solution, f1_value = self.greedy_algorithm.run()
        f2_value = self.f2(solution)
        f3_value = self.f3(solution)

        self.add_solutions(solution, f1_value, f2_value, f3_value)

        solution, f1_value = self.local_search(solution, f1_value)
        f2_value = self.f2(solution)
        f3_value = self.f3(solution)

        self.add_solutions(solution, f1_value, f2_value, f3_value)

        return

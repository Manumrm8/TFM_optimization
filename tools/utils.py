import random
import numpy as np
from scipy.spatial import distance
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def fitness(individual, df_proveer, df_suministro):
    max_dist = 0
    for _, sitio in df_proveer.iterrows():
        min_dist = float("inf")
        for idx in individual:
            punto = df_suministro.iloc[idx]
            dist = distance.euclidean(sitio, punto)
            min_dist = min(min_dist, dist)
        if max(max_dist, min_dist) == min_dist:
            max_dist = min_dist
    return max_dist


def create_population(df_suministro, population_size, k):
    return [random.sample(range(len(df_suministro)), k) for _ in range(population_size)]


def selection(population, df_proveer, df_suministro):
    # Usamos tqdm para mostrar el progreso al calcular la fitness
    fitness_values = [
        (fitness(individuo, df_proveer, df_suministro), individuo)
        for individuo in tqdm(
            population, desc="Calculando fitness de la población", leave=False
        )
    ]

    # Ordenar por fitness
    population_sorted = sorted(fitness_values, key=lambda x: x[1])

    # Devolvemos los mejores individuos
    return [individuo for _, individuo in population_sorted[: len(population) // 2]]


def crossover(parent1, parent2, k):
    split = random.randint(1, k - 1)
    child1 = parent1[:split] + [gene for gene in parent2 if gene not in parent1[:split]]
    child2 = parent2[:split] + [gene for gene in parent1 if gene not in parent2[:split]]
    return child1[:k], child2[:k]


def mutate(individual, mutation_rate, df_suministro, k):
    if random.random() < mutation_rate:
        idx_to_mutate = random.randint(0, k - 1)
        possible_genes = [i for i in range(len(df_suministro)) if i not in individual]

        # Verificación para evitar índices fuera de rango
        if possible_genes:
            new_gene = random.choice(possible_genes)
            individual[idx_to_mutate] = new_gene


def genetic_algorithm(
    df_suministro,
    df_proveer,
    k,
    num_generations,
    population_size,
    mutation_rate,
):
    population = create_population(df_suministro, population_size, k)
    best_solution = None
    best_fitness = float("inf")

    for _ in tqdm(range(num_generations), desc="Generaciones"):
        # Evaluación y selección
        population = selection(population, df_proveer, df_suministro)

        # Guardar el mejor individuo
        current_best = population[0]
        current_fitness = fitness(current_best, df_proveer, df_suministro)
        if current_fitness < best_fitness:
            best_fitness = current_fitness
            print(best_fitness)
            best_solution = current_best

        # Crear nueva generación
        new_population = population[:]
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population, 2)
            child1, child2 = crossover(parent1, parent2, k)
            mutate(child1, mutation_rate, df_suministro, k)
            mutate(child2, mutation_rate, df_suministro, k)

            # Verificación de índices para evitar valores fuera de rango
            if all(gene < len(df_suministro) for gene in child1):
                new_population.append(child1)
            if all(gene < len(df_suministro) for gene in child2):
                new_population.append(child2)

        population = new_population

    return best_solution, best_fitness


def plot_results(best_solution, df_proveer, df_suministro):
    plt.figure(figsize=(10, 8))

    # Dibujar sitios a proveer
    plt.scatter(
        df_proveer.iloc[:, 0],
        df_proveer.iloc[:, 1],
        c="blue",
        label="Sitios a Proveer",
        s=100,
    )

    # Dibujar todos los puntos de suministro
    plt.scatter(
        df_suministro.iloc[:, 0],
        df_suministro.iloc[:, 1],
        c="gray",
        label="Puntos de Suministro",
        s=50,
        alpha=0.5,
    )

    # Dibujar puntos de suministro seleccionados
    selected_points = df_suministro.iloc[best_solution]
    plt.scatter(
        selected_points.iloc[:, 0],
        selected_points.iloc[:, 1],
        c="red",
        label="Puntos Seleccionados",
        s=100,
    )

    # Personalizar el gráfico
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.legend()
    plt.title("Distribución de Sitios y Puntos de Suministro")
    plt.grid()
    plt.show()

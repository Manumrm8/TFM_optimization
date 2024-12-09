import random
import matplotlib.pyplot as plt
from tqdm import tqdm


def fitness(individual, df_proveer, df_distances):
    """Calcula la distancia del elemento más lejano a los puntos de suministro."""
    dist = []
    for sitio in df_proveer.index:
        distancia = df_distances.iloc[sitio, individual].min()
        dist.append(distancia)
    return max(dist)


def create_population(df_suministro, population_size, k):
    """Crea una población inicial de individuos aleatorios."""
    return [random.sample(range(len(df_suministro)), k) for _ in range(population_size)]


def selection(population, df_proveer, df_distances):
    """Selecciona los mejores individuos de la población."""
    # Usamos tqdm para mostrar el progreso al calcular la fitness
    fitness_values = [
        (fitness(individuo, df_proveer, df_distances), individuo)
        for individuo in tqdm(
            population, desc="Calculando fitness de la población", leave=False
        )
    ]

    # Ordenar por fitness
    population_sorted = sorted(fitness_values, key=lambda x: x[1])

    selected_population = [
        individuo for _, individuo in population_sorted[: len(population) // 2]
    ]

    # Obtener el mejor individuo y su valor de fitness
    best_fitness, best_individual = population_sorted[0]

    # Devolver la población seleccionada y el mejor individuo con su fitness
    return selected_population, (best_fitness, best_individual)


def crossover(parent1, parent2, k):
    """Cruza dos individuos para crear dos nuevos individuos."""
    split = random.randint(1, k - 1)
    child1 = parent1[:split] + [gene for gene in parent2 if gene not in parent1[:split]]
    child2 = parent2[:split] + [gene for gene in parent1 if gene not in parent2[:split]]
    return child1[:k], child2[:k]


# Mutación
def mutate(individual, mutation_rate, df_suministro, k):
    """Realiza una mutación en un individuo."""
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
    df_distances,
    k,
    num_generations,
    population_size,
    mutation_rate,
):
    """
    Implementación de un algoritmo genético para resolver el problema de ubicación de k centros
    balanceados.

    Parámetros:
    df_suministro (pd.DataFrame): DataFrame con los datos de los puntos de suministro.
    df_proveer (pd.DataFrame): DataFrame con los datos de los puntos a proveer.
    df_distances (pd.DataFrame): DataFrame con la matriz de distancias entre los puntos de suministro
    y los puntos a proveer.
    k (int): Número de centros a ubicar.
    num_generations (int): Número de generaciones del algoritmo.
    population_size (int): Tamaño de la población.
    mutation_rate (float): Tasa de mutación.

    Devuelve:
    tuple: Mejor individuo y su fitness.
    """

    population = create_population(df_suministro, population_size, k)
    best_solution = None
    best_fitness = float("inf")

    for _ in tqdm(range(num_generations), desc="Generaciones"):
        # Evaluación y selección
        selected_population, best = selection(population, df_proveer, df_distances)

        # Actualizar la población con los seleccionados
        population = selected_population

        # Guardar el mejor individuo y su fitness
        current_best = best[1]  # El mejor individuo
        current_fitness = best[0]

        print(f"actualmente, la mejor solucion es: {current_fitness}")
        if current_fitness < best_fitness:
            best_fitness = current_fitness
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

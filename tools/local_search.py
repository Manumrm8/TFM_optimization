import numpy as np

def local_search_optimized(supply_selected, df_distances_demand, k, m):
    """
    Realiza una búsqueda local optimizada integrando las funciones objetivo.

    Utiliza arrays de NumPy para acelerar los cálculos y una estrategia de "primer mejor",
    aceptando el primer intercambio que domine la solución actual y reiniciando la búsqueda.

    Args:
        - supply_selected (list): La solución inicial.
        - df_distances_demand (pd.DataFrame): DataFrame de distancias.
        - k (int): Número de elementos en la solución.
        - m (int): Número total de posibles instalaciones.

    Returns:
        - list: La mejor solución encontrada.
    """
    # 1. Convertir a NumPy para un rendimiento mucho mayor.
    all_distances = df_distances_demand.values
    new_supply = list(supply_selected)

    while True:
        improvement_found = False

        # 2. Calcular los valores objetivo de la solución actual UNA SOLA VEZ por iteración.
        current_supply_distances = all_distances[:, new_supply]
        
        # Calcular asignaciones y distancias mínimas para la solución actual
        min_distances = current_supply_distances.min(axis=1)
        assignment_indices = current_supply_distances.argmin(axis=1)
        
        # Calcular f1
        f1_current = min_distances.max()
        
        # Calcular f2 y f3
        _, counts = np.unique(assignment_indices, return_counts=True)
        
        # Si un centro no recibe asignaciones, su conteo será 0. Debemos considerarlo.
        # Creamos un array de ceros para todos los `k` centros y lo llenamos con los conteos.
        all_counts = np.zeros(k, dtype=int)
        all_counts[:len(counts)] = counts
        
        f2_current = all_counts.max()
        f3_current = all_counts.max() - all_counts.min()

        # Iterar para encontrar un vecino que mejore la solución
        for i in range(k):
            s_out = new_supply[i] # No se utiliza pero está aquí para que sea más legible
            
            for s_in in range(m):
                # El candidato a entrar no puede estar ya en la solución
                if s_in in new_supply:
                    continue

                # 3. Crear una solución vecina y calcular sus objetivos eficientemente
                temp_supply = new_supply[:]
                temp_supply[i] = s_in
                
                temp_supply_distances = all_distances[:, temp_supply]
                
                # Reutilizar cálculos para obtener f1, f2, f3 del vecino
                min_distances_temp = temp_supply_distances.min(axis=1)
                assignment_indices_temp = temp_supply_distances.argmin(axis=1)
                
                f1_new = min_distances_temp.max()
                
                _, counts_new = np.unique(assignment_indices_temp, return_counts=True)
                all_counts_new = np.zeros(k, dtype=int)
                all_counts_new[:len(counts_new)] = counts_new
                
                f2_new = all_counts_new.max()
                f3_new = f2_new - all_counts_new.min()
                
                # 4. Comprobar dominancia
                is_dominant = (f1_new <= f1_current and f2_new <= f2_current and f3_new <= f3_current) and \
                              (f1_new < f1_current or f2_new < f2_current or f3_new < f3_current)

                if is_dominant:
                    # 5. Si se encuentra mejora, actualizar la solución y reiniciar el bucle principal
                    new_supply = temp_supply
                    improvement_found = True
                    break # Salir del bucle de s_in
            
            if improvement_found:
                break # Salir del bucle de s_out
        
        # Si un ciclo completo no arrojó mejoras, hemos alcanzado un óptimo local.
        if not improvement_found:
            break

    return new_supply


def local_search_f1_optimized(supply_selected, df_distances_demand, k, m, n_veces):
    """
    Realiza una búsqueda local optimizada para f1 (minimizar la máxima distancia).
    Utiliza una estrategia de "primer mejor" y NumPy para mayor rendimiento.

    Args:
        supply_selected (list): Solución inicial.
        df_distances_demand (pd.DataFrame): DataFrame de distancias.
        k (int): Número de elementos en la solución.
        m (int): Número total de posibles instalaciones.
        n_veces (int): Límite de mejoras a realizar antes de detenerse.

    Returns:
        tuple: (mejor_solucion, mejor_valor_objetivo)
    """
    # 1. Convertir a NumPy para cálculos mucho más rápidos dentro de los bucles.
    all_distances = df_distances_demand.values
    
    current_solution = list(supply_selected)
    
    # Calcular el valor objetivo inicial
    current_objective = np.max(np.min(all_distances[:, current_solution], axis=1))
    
    improvements_count = 0

    while True:
        improvement_found_in_pass = False
        
        # Iterar sobre las posiciones de la solución para intercambiar
        for i in range(k):
            # Iterar sobre todos los posibles candidatos a entrar en la solución
            for j in range(m):
                # El candidato no puede estar ya en la solución
                if j in current_solution:
                    continue
                
                # 2. Crear una solución vecina (temporal)
                temp_solution = current_solution[:]
                temp_solution[i] = j
                
                # 3. Calcular el objetivo de la solución vecina (¡CORREGIDO Y OPTIMIZADO!)
                temp_objective = np.max(np.min(all_distances[:, temp_solution], axis=1))
                
                # 4. Comprobar si hay una mejora
                if temp_objective < current_objective:
                    # Aplicar la mejora y reiniciar la búsqueda desde este nuevo punto
                    current_solution = temp_solution
                    current_objective = temp_objective
                    
                    improvement_found_in_pass = True
                    improvements_count += 1
                    
                    # Condición de parada por número de mejoras
                    if improvements_count == n_veces:
                        return current_solution
                    
                    # Romper los bucles para reiniciar la búsqueda con la nueva solución
                    break  # Salir del bucle de 'j'
            
            if improvement_found_in_pass:
                break  # Salir del bucle de 'i'
        
        # 5. Si se completa un ciclo entero sin encontrar mejoras, la búsqueda termina.
        if not improvement_found_in_pass:
            break

    return current_solution


def local_search_f2_optimized(supply_selected, df_distances_demand, k, m, n_veces):
    """
    Realiza una búsqueda local optimizada para f2 (minimizar la demanda máxima por centro).
    Utiliza una estrategia de "primer mejor" y NumPy para mayor rendimiento.

    Args:
        solution (list): Solución inicial.
        df_distances_demand (pd.DataFrame): DataFrame de distancias.
        k (int): Número de elementos en la solución.
        m (int): Número total de posibles instalaciones.
        n_veces (int): Límite de mejoras a realizar antes de detenerse.

    Returns:
        tuple: (mejor_solucion, mejor_valor_objetivo)
    """
    # 1. Convertir a NumPy para cálculos mucho más rápidos.
    all_distances = df_distances_demand.values
    
    current_solution = list(supply_selected)
    
    # Calcular el valor objetivo inicial de forma eficiente
    assignment_indices = np.argmin(all_distances[:, current_solution], axis=1)
    # np.unique con return_counts=True es el equivalente optimizado de value_counts()
    _, counts = np.unique(assignment_indices, return_counts=True)
    current_objective = np.max(counts)
    
    improvements_count = 0

    while True:
        improvement_found_in_pass = False
        
        # Iterar sobre las posiciones de la solución para intercambiar
        for i in range(k):
            # Iterar sobre todos los posibles candidatos a entrar
            for j in range(m):
                # El candidato no puede estar ya en la solución
                if j in current_solution:
                    continue
                
                # 2. Crear una solución vecina (temporal)
                temp_solution = current_solution[:]
                temp_solution[i] = j
                
                # 3. Calcular el objetivo de la solución vecina (¡CORREGIDO Y OPTIMIZADO!)
                temp_assignment_indices = np.argmin(all_distances[:, temp_solution], axis=1)
                _, temp_counts = np.unique(temp_assignment_indices, return_counts=True)
                temp_objective = np.max(temp_counts)
                
                # 4. Comprobar si hay una mejora
                if temp_objective < current_objective:
                    # Aplicar la mejora y reiniciar la búsqueda desde este nuevo punto
                    current_solution = temp_solution
                    current_objective = temp_objective
                    
                    improvement_found_in_pass = True
                    improvements_count += 1
                    
                    # Condición de parada por número de mejoras
                    if improvements_count == n_veces:
                        return current_solution
                    
                    # Romper los bucles para reiniciar la búsqueda
                    break  # Salir del bucle de 'j'
            
            if improvement_found_in_pass:
                break  # Salir del bule de 'i'
        
        # 5. Si no se encuentran mejoras en una pasada completa, la búsqueda termina.
        if not improvement_found_in_pass:
            break

    return current_solution


def local_search_f3_optimized(supply_selected, df_distances_demand, k, m, n_veces):
    """
    Realiza una búsqueda local optimizada para f3 (minimizar max_carga - min_carga).
    Utiliza una estrategia de "primer mejor" y maneja correctamente los conteos cero.

    Args:
        solution (list): Solución inicial.
        df_distances_demand (pd.DataFrame): DataFrame de distancias.
        k (int): Número de elementos en la solución.
        m (int): Número total de posibles instalaciones.
        n_veces (int): Límite de mejoras a realizar antes de detenerse.

    Returns:
        tuple: (mejor_solucion, mejor_valor_objetivo)
    """
    # 1. Convertir a NumPy para cálculos mucho más rápidos.
    all_distances = df_distances_demand.values
    
    current_solution = list(supply_selected)
    
    # --- Calcular el valor objetivo inicial de forma eficiente ---
    assignment_indices = np.argmin(all_distances[:, current_solution], axis=1)
    unique_indices, counts = np.unique(assignment_indices, return_counts=True)
    
    # 2. Manejo de conteos cero: crucial para f3.
    # Creamos un array de ceros para los k centros.
    all_counts = np.zeros(k, dtype=int)
    # Rellenamos solo los que tienen asignaciones. Los demás se quedan en cero.
    all_counts[unique_indices] = counts
    
    current_objective = np.max(all_counts) - np.min(all_counts)
    
    improvements_count = 0

    while True:
        improvement_found_in_pass = False
        
        for i in range(k):
            for j in range(m):
                if j in current_solution:
                    continue
                
                # Crear una solución vecina (temporal)
                temp_solution = current_solution[:]
                temp_solution[i] = j
                
                # --- Calcular el objetivo de la solución vecina (¡CORREGIDO Y OPTIMIZADO!) ---
                temp_assignment_indices = np.argmin(all_distances[:, temp_solution], axis=1)
                temp_unique_indices, temp_counts = np.unique(temp_assignment_indices, return_counts=True)
                
                temp_all_counts = np.zeros(k, dtype=int)
                temp_all_counts[temp_unique_indices] = temp_counts
                
                temp_objective = np.max(temp_all_counts) - np.min(temp_all_counts)
                
                # 4. Comprobar si hay una mejora
                if temp_objective < current_objective:
                    # Aplicar la mejora y reiniciar la búsqueda
                    current_solution = temp_solution
                    current_objective = temp_objective
                    
                    improvement_found_in_pass = True
                    improvements_count += 1
                    
                    if improvements_count == n_veces:
                        return current_solution
                    
                    break  # Salir del bucle de 'j'
            
            if improvement_found_in_pass:
                break  # Salir del bucle de 'i'
        
        # 5. Si no hay mejoras en una pasada, la búsqueda termina.
        if not improvement_found_in_pass:
            break

    return current_solution
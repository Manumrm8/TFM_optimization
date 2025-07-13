# Funcion para calcular la distancia minima de los puntos de suministro a los puntos de demanda.
def max_min_dist(df_distances_demand, supply_selected):
    """
    - df_distances_demand: DataFrame con las distancias entre los puntos de suministro y los puntos de demanda.
    - supply_selected: Lista con los indices de los puntos de suministro seleccionados.
    """

    dist = []
    for demand_point in df_distances_demand.index:
        distancia = df_distances_demand.iloc[demand_point, supply_selected].min()
        dist.append(distancia)
    return max(dist)


# Función para calcular el número máximo de puntos de demanda asignados a un punto de suministro
def max_supply_demanded(df_distances_demand, supply_selected):
    asignacion = df_distances_demand.iloc[:, supply_selected].idxmin(axis=1)
    maximum = asignacion.value_counts().max()
    return maximum


# Función para calcular el número máximo de puntos de demanda asignados a un punto de suministro - el mínimo.
def balanced_supply_demanded(df_distances_demand, supply_selected):
    asignacion = df_distances_demand.iloc[:, supply_selected].idxmin(axis=1)
    maximum = asignacion.value_counts().max()
    minimum = asignacion.value_counts().min()
    return maximum - minimum


def OFWeight(df_distances_demand, supply_selected, weight=[1, 1, 1]):

    value = weight[0] * max_min_dist(df_distances_demand, supply_selected)
    +weight[1] * max_supply_demanded(df_distances_demand, supply_selected)
    +weight[2] * balanced_supply_demanded(df_distances_demand, supply_selected)
    return value

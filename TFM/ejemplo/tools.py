import folium
import random
import math


def generar_coordenadas_aleatorias(
    n_puntos, centro_latitud, centro_longitud, radio_dispersion_km, random_seed=None
):
    """
    Genera una lista de coordenadas (latitud, longitud) aleatorias
    dentro de un radio dado desde un centro.

    Args:
        n_puntos (int): Número de puntos a generar.
        centro_latitud (float): Latitud del centro.
        centro_longitud (float): Longitud del centro.
        radio_dispersion_km (int): Radio de dispersión en kilómetros.
        random_seed (int, optional): Semilla para el generador de números aleatorios.
                                     Si se proporciona, los resultados serán reproducibles.
                                     Por defecto es None (comportamiento aleatorio).
    """
    if random_seed is not None:
        random.seed(random_seed)

    coordenadas = []

    def get_random_point_in_radius(lat_center, lon_center, radius_km):
        theta = random.uniform(0, 2 * math.pi)
        r = radius_km * (random.random() ** 0.5)

        delta_x_km = r * math.cos(theta)
        delta_y_km = r * math.sin(theta)

        new_lat = lat_center + (delta_y_km / 111.0)

        lon_conversion_factor = 111.0 * math.cos(math.radians(lat_center))
        if lon_conversion_factor == 0:
            new_lon = lon_center
        else:
            new_lon = lon_center + (delta_x_km / lon_conversion_factor)

        return new_lat, new_lon

    for _ in range(n_puntos):
        lat, lon = get_random_point_in_radius(
            centro_latitud, centro_longitud, radio_dispersion_km
        )
        coordenadas.append((lat, lon))
    return coordenadas


def crear_mapa_con_puntos(
    n_azules,
    m_verdes,
    k_rojos,
    coord_azules,
    coord_verdes_originales,
    centro_latitud,
    centro_longitud,
    zoom_nivel,
    nombre_mapa,
    random_seed_rojos=None,
):
    """
    Crea y guarda un mapa de Folium con los puntos especificados,
    pintando 'k_rojos' de los puntos verdes originales de rojo.

    Args:
        n_azules (int): Número de puntos azules (usado para tooltips, no para generación).
        m_verdes (int): Número total de puntos verdes (usado para tooltips, no para generación).
        k_rojos (int): Número de puntos a seleccionar de los 'm' verdes para pintar de rojo.
        coord_azules (list): Lista de tuplas (lat, lon) para los puntos azules.
        coord_verdes_originales (list): Lista de tuplas (lat, lon) para los puntos verdes.
        centro_latitud (float): Latitud para centrar el mapa.
        centro_longitud (float): Longitud para centrar el mapa.
        zoom_nivel (int): Nivel de zoom inicial del mapa.
        nombre_mapa (str): Nombre del archivo HTML a guardar.
        random_seed_rojos (int, optional): Semilla para la selección aleatoria de puntos rojos.
                                           Si se proporciona, la selección de rojos será reproducible.
                                           Por defecto es None.
    """
    if k_rojos > len(coord_verdes_originales):
        print(
            f"Advertencia: k_rojos ({k_rojos}) es mayor que el número de puntos verdes ({len(coord_verdes_originales)})."
        )
        print(
            f"Se pintarán todos los {len(coord_verdes_originales)} puntos verdes como rojos."
        )
        k_rojos = len(coord_verdes_originales)

    # Establecer la semilla para la selección de rojos si se proporciona
    if random_seed_rojos is not None:
        random.seed(random_seed_rojos)

    m = folium.Map(location=[centro_latitud, centro_longitud], zoom_start=zoom_nivel)

    # Añadir puntos azules
    for i, (lat, lon) in enumerate(coord_azules):
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.7,
            tooltip=f"Punto Azul {i+1}",
        ).add_to(m)

    # Seleccionar k puntos para ser rojos de los verdes originales
    # random.sample requiere que la población sea mayor o igual a k.
    if k_rojos > 0:
        puntos_rojos_seleccionados = random.sample(coord_verdes_originales, k_rojos)
        # Convertir a un conjunto para búsquedas eficientes
        # Nota: La tupla (lat, lon) puede no ser un hashable perfecto si los flotantes tienen precisiones minúsculas que difieren
        # Para evitar problemas con la comparación de floats, a veces es mejor redondear o usar un enfoque diferente.
        # Sin embargo, para coordenadas recién generadas así, debería funcionar bien.
        set_puntos_rojos = set(puntos_rojos_seleccionados)
    else:
        set_puntos_rojos = set()

    # Añadir los puntos verdes y rojos
    for i, (lat, lon) in enumerate(coord_verdes_originales):
        if (lat, lon) in set_puntos_rojos:
            # Este punto es uno de los k seleccionados para ser rojo
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.7,
                tooltip=f"Punto Rojo (originalmente Verde) {i+1}",
            ).add_to(m)
        else:
            # Este punto se mantiene verde
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color="green",
                fill=True,
                fill_color="green",
                fill_opacity=0.7,
                tooltip=f"Punto Verde {i+1}",
            ).add_to(m)

    m.save(nombre_mapa)
    print(f"Mapa guardado en {nombre_mapa}")
    print(f"Abre '{nombre_mapa}' en tu navegador para verlo.")

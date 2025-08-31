import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def contar_soluciones_en_pareto(archive: str, total_corridas: int = 110) -> list:
    """
    Carga un frente de Pareto desde un CSV y cuenta cuántas soluciones de cada
    archivo de resultados (también CSV) se encuentran en dicho frente, usando
    solo las columnas 'f1', 'f2' y 'f3'.

    Args:
        archive (str): El nombre base del archivo para construir las rutas de soluciones.
        total_corridas (int): El número total de archivos de soluciones a procesar.

    Returns:
        list: Una lista donde cada elemento es el recuento de soluciones
              en el frente de Pareto para el archivo correspondiente.
    """
    path_pareto = f"Pareto_front/{archive}.csv"
    path_soluciones = f"Solutions/Multiprocessing/{archive}"

    # 1. Cargar el frente de Pareto de referencia desde un CSV
    try:
        # Leer el archivo CSV completo
        df_pareto = pd.read_csv(path_pareto)
        # Seleccionar solo las columnas de interés y convertir a un array de NumPy
        frente_pareto = df_pareto[["f1", "f2", "f3"]].values

        print(
            f"✅ Frente de Pareto cargado desde '{path_pareto}'. Contiene {len(frente_pareto)} soluciones."
        )
    except FileNotFoundError:
        print(
            f"❌ Error: No se encontró el archivo del frente de Pareto en '{path_pareto}'."
        )
        return []
    except KeyError:
        print(
            f"❌ Error: El archivo '{path_pareto}' no contiene las columnas 'f1', 'f2' y 'f3'."
        )
        return []
    except Exception as e:
        print(f"❌ Error al leer el archivo del frente de Pareto: {e}")
        return []

    # 2. Iterar a través de cada archivo de solución y contar las coincidencias
    conteo_por_archivo = []

    for i in range(1, total_corridas + 1):
        # Construir la ruta para cada archivo CSV de solución
        ruta_solution = f"{path_soluciones}/{archive}_#{i}.csv"

        if not os.path.exists(ruta_solution):
            print(
                f"⚠️  Advertencia: El archivo '{ruta_solution}' no existe. Se contará como 0."
            )
            conteo_por_archivo.append(0)
            continue

        try:
            # Cargar las soluciones del archivo actual
            df_soluciones = pd.read_csv(ruta_solution)
            # Seleccionar solo las columnas de interés y convertir a NumPy array
            soluciones_actuales = df_soluciones[["f1", "f2", "f3"]].values

            # Contar cuántas soluciones están en el frente de Pareto
            contador = 0
            if (
                soluciones_actuales.size > 0
            ):  # Asegurarse de que no está vacío tras seleccionar
                for sol in soluciones_actuales:
                    if any(
                        np.allclose(sol, p_sol, atol=1e-6) for p_sol in frente_pareto
                    ):
                        contador += 1

            conteo_por_archivo.append(contador)

        except pd.errors.EmptyDataError:
            print(
                f"ℹ️  Info: El archivo '{ruta_solution}' está vacío. Se contará como 0."
            )
            conteo_por_archivo.append(0)
        except KeyError:
            print(
                f"⚠️  Advertencia: El archivo '{ruta_solution}' no contiene las columnas 'f1', 'f2' y 'f3'. Se contará como 0."
            )
            conteo_por_archivo.append(0)
        except Exception as e:
            print(f"❌ Error procesando el archivo '{ruta_solution}': {e}")
            conteo_por_archivo.append(0)

    return conteo_por_archivo


def graficar_boxplots(vector_de_conteos: list, archive: str):
    """
    Genera una gráfica de boxplots a partir de un vector de conteos.

    La función asume que el vector contiene 220 elementos que corresponden a
    11 grupos de 20 ejecuciones cada uno, donde cada grupo representa un
    valor de beta de 0.0 a 1.0.

    Args:
        vector_de_conteos (list): La lista de 220 conteos de soluciones.
        archive (str): El nombre base del archivo, usado para el título del gráfico.
    """
    if len(vector_de_conteos) != 220:
        print(
            f"❌ Error: Se esperaba un vector con 220 elementos, pero se recibieron {len(vector_de_conteos)}."
        )
        return

    # 1. Preparar los datos para la gráfica
    # Creamos las etiquetas para los 11 grupos de beta
    betas = np.linspace(0, 1, 11)  # [0.0, 0.1, ..., 1.0]

    # Repetimos cada valor de beta 20 veces para que coincida con cada conteo
    grupos_beta = np.repeat(betas, 20)

    # Creamos un DataFrame de Pandas, que es el formato ideal para Seaborn
    df_grafica = pd.DataFrame(
        {"Beta": grupos_beta, "Conteo de Soluciones": vector_de_conteos}
    )

    df_grafica["Beta"] = df_grafica["Beta"].map("{:.1f}".format)

    # 2. Crear la gráfica
    sns.set_theme(style="whitegrid")  # Establecer un estilo visual agradable
    plt.figure(figsize=(14, 8))  # Definir el tamaño de la figura

    # Crear el boxplot
    ax = sns.boxplot(
        x="Beta",
        y="Conteo de Soluciones",
        data=df_grafica,
        palette="viridis",
        hue="Beta",
        legend=False,
    )

    # 3. Personalizar la gráfica
    plt.title(
        f"Distribución de Soluciones Encontradas en el Frente de Pareto\n(Instancia: {archive})",
        fontsize=16,
    )
    plt.xlabel("Valor de Alpha", fontsize=12)
    plt.ylabel("Número de Soluciones en el Frente", fontsize=12)

    # ---> MODIFICACIÓN: Establecer los límites del eje Y <---
    plt.ylim(0, 18)

    plt.tight_layout()  # Ajusta el gráfico para que todo encaje bien

    # 4. Mostrar la gráfica
    plt.show()


def graficar_boxplots_comparativo(
    conteos_modelo1: list,
    conteos_modelo2: list,
    conteos_modelo3: list,
    nombres_modelos: list,
    archive: str,
):
    """
    Genera una gráfica de boxplots comparando los resultados de tres modelos.

    La función asume que cada vector contiene 220 elementos que corresponden a
    11 grupos de 20 ejecuciones cada uno (para valores de beta de 0.0 a 1.0).

    Args:
        conteos_modelo1 (list): Vector de conteos del primer modelo.
        conteos_modelo2 (list): Vector de conteos del segundo modelo.
        conteos_modelo3 (list): Vector de conteos del tercer modelo.
        nombres_modelos (list): Una lista con tres strings para los nombres
                                de los modelos. Ej: ['Modelo A', 'Modelo B', 'Modelo C'].
        archive (str): El nombre base del archivo, usado para el título.
    """
    # 1. Validar las entradas
    if not all(len(v) == 220 for v in [conteos_modelo1, conteos_modelo2, conteos_modelo3]):
        print("❌ Error: Todos los vectores de conteos deben tener 220 elementos.")
        return
    if len(nombres_modelos) != 3:
        print("❌ Error: Se debe proporcionar una lista con exactamente 3 nombres de modelos.")
        return

    # 2. Preparar los datos para la gráfica
    betas = np.linspace(0, 1, 11)  # [0.0, 0.1, ..., 1.0]
    grupos_beta = np.repeat(betas, 20)

    # Crear un DataFrame para cada modelo y luego unirlos
    df1 = pd.DataFrame({
        "Alpha": grupos_beta,
        "Conteo de Soluciones": conteos_modelo1,
        "Número de iteraciones": nombres_modelos[0]
    })
    df2 = pd.DataFrame({
        "Alpha": grupos_beta,
        "Conteo de Soluciones": conteos_modelo2,
        "Número de iteraciones": nombres_modelos[1]
    })
    df3 = pd.DataFrame({
        "Alpha": grupos_beta,
        "Conteo de Soluciones": conteos_modelo3,
        "Número de iteraciones": nombres_modelos[2]
    })

    # Combinar los tres DataFrames en uno solo
    df_grafica = pd.concat([df1, df2, df3], ignore_index=True)
    
    # Formatear la columna Beta para que se muestre con un solo decimal
    df_grafica["Alpha"] = df_grafica["Alpha"].map("{:.1f}".format)


    # 3. Crear la gráfica
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(18, 9)) # Aumentamos el tamaño para mejor legibilidad

    # Crear el boxplot usando 'hue' para diferenciar los modelos
    ax = sns.boxplot(
        x="Alpha",
        y="Conteo de Soluciones",
        hue="Número de iteraciones",  # <-- La clave para la comparación
        data=df_grafica,
        palette="viridis",
    )

    # 4. Personalizar la gráfica
    plt.title(
        f"Compativa de resultados en función del Alpha\n(Instancia: {archive})",
        fontsize=18,
    )
    plt.xlabel("Valor de Alpha", fontsize=14)
    plt.ylabel("Número de Soluciones en el Frente", fontsize=14)
    
    # Establecer el límite del eje Y como en la función anterior
    plt.ylim(0, 18)
    
    # Mejorar la leyenda
    plt.legend(title="Número de iteraciones", fontsize=12)
    
    plt.tight_layout()

    # 5. Mostrar la gráfica
    plt.show()

def borrar_resultados_hashtag(ruta_carpeta):
    """
    Elimina todos los archivos dentro de una carpeta específica si su nombre
    contiene el carácter '#'.

    :param ruta_carpeta: La ruta a la carpeta donde se buscarán los archivos.
    """
    # Comprobar si la ruta de la carpeta existe
    if not os.path.isdir(ruta_carpeta):
        print(f"Error: La carpeta '{ruta_carpeta}' no existe.")
        return

    print(f"Buscando archivos con '#' en la carpeta: {ruta_carpeta}")

    # Recorrer todos los elementos en el directorio
    for nombre_archivo in os.listdir(ruta_carpeta):
        # Comprobar si el '#' está en el nombre del archivo
        if "#" in nombre_archivo:
            # Construir la ruta completa del archivo
            ruta_completa = os.path.join(ruta_carpeta, nombre_archivo)

            # Asegurarse de que es un archivo y no una carpeta
            if os.path.isfile(ruta_completa):
                try:
                    # Eliminar el archivo
                    os.remove(ruta_completa)
                except OSError as e:
                    print(f"Error al eliminar el archivo {nombre_archivo}: {e}")

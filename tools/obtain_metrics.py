import pandas as pd
import numpy as np
import os

def calcular_coverage_pareto(archive: str, total_corridas: int = 110) -> list:
    """
    Calcula la métrica de cobertura (coverage) de un frente de Pareto de referencia
    sobre un conjunto de archivos de soluciones.

    La métrica se define como la proporción de soluciones en un archivo que son
    dominadas por al menos una solución en el frente de Pareto de referencia.
    Se utilizan únicamente las columnas 'f1', 'f2' y 'f3' para la comparación.

    Args:
        archive (str): El nombre base del archivo para construir las rutas.
        total_corridas (int): El número total de archivos de soluciones a procesar.

    Returns:
        list: Una lista de flotantes donde cada elemento es el valor de coverage
              (entre 0.0 y 1.0) para el archivo de solución correspondiente.
    """
    path_pareto = f"Pareto_front/{archive}.csv"
    path_soluciones = f"Solutions/Multiprocessing/{archive}"

    # 1. Cargar el frente de Pareto de referencia desde un CSV
    try:
        df_pareto = pd.read_csv(path_pareto)
        # Seleccionar solo las columnas de interés y convertir a un array de NumPy
        frente_pareto = df_pareto[["f1", "f2", "f3"]].values
        print(
            f"✅ Frente de Pareto cargado desde '{path_pareto}'. Contiene {len(frente_pareto)} soluciones."
        )
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo del frente de Pareto en '{path_pareto}'.")
        return []
    except KeyError:
        print(f"❌ Error: El archivo '{path_pareto}' no contiene las columnas 'f1', 'f2' y 'f3'.")
        return []
    except Exception as e:
        print(f"❌ Error al leer el archivo del frente de Pareto: {e}")
        return []

    # 2. Iterar a través de cada archivo de solución para calcular el coverage
    coverage_por_archivo = []

    for i in range(1, total_corridas + 1):
        ruta_solution = f"{path_soluciones}/{archive}_#{i}.csv"

        if not os.path.exists(ruta_solution):
            print(f"⚠️  Advertencia: El archivo '{ruta_solution}' no existe. Se asigna coverage de 0.0.")
            coverage_por_archivo.append(0.0)
            continue

        try:
            df_soluciones = pd.read_csv(ruta_solution)
            soluciones_actuales = df_soluciones[["f1", "f2", "f3"]].values
            
            total_soluciones_archivo = len(soluciones_actuales)

            # Si el archivo de soluciones está vacío, el coverage es 0
            if total_soluciones_archivo == 0:
                coverage_por_archivo.append(0.0)
                continue
            
            # --- Lógica de cálculo de dominancia (vectorizada para eficiencia) ---
            # Se asume que valores más bajos son mejores (minimización)
            
            # Comparamos cada solución actual con cada solución del frente de Pareto
            # all_le determina si una solución del frente es <= en todos los objetivos
            # any_l determina si una solución del frente es < en al menos un objetivo
            all_le = np.all(frente_pareto[:, np.newaxis, :] <= soluciones_actuales[np.newaxis, :, :], axis=2)
            any_l = np.any(frente_pareto[:, np.newaxis, :] < soluciones_actuales[np.newaxis, :, :], axis=2)
            
            # Una solución es dominada si existe al menos una en el frente que cumple ambas condiciones
            dominada_mask = np.any(all_le & any_l, axis=0)
            
            # Contamos cuántas soluciones del archivo actual son dominadas
            num_dominadas = np.sum(dominada_mask)
            
            # Calculamos el coverage
            coverage = num_dominadas / total_soluciones_archivo
            coverage_por_archivo.append(coverage)

        except pd.errors.EmptyDataError:
            print(f"ℹ️  Info: El archivo '{ruta_solution}' está vacío. Se asigna coverage de 0.0.")
            coverage_por_archivo.append(0.0)
        except KeyError:
            print(f"⚠️  Advertencia: El archivo '{ruta_solution}' no contiene las columnas requeridas. Se asigna coverage de 0.0.")
            coverage_por_archivo.append(0.0)
        except Exception as e:
            print(f"❌ Error procesando el archivo '{ruta_solution}': {e}")
            coverage_por_archivo.append(0.0)

    return coverage_por_archivo
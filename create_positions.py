import random


def procesar_archivo(entrada, salida, semilla=42):
    with open(entrada, "r") as file:
        lineas = file.readlines()

    # Leer los valores de la primera línea
    x, y, _ = map(int, lineas[0].split())

    # Obtener las coordenadas de los puntos a suministrar y fábricas
    puntos_suministro = lineas[1 : x + 1]  # Desde la segunda línea hasta x
    fabricas = lineas[x + 1 : x + 1 + y]  # Desde x+1 hasta x+1+y

    # Fijar semilla para reproducibilidad
    random.seed(semilla)

    # Muestreo aleatorio sin reemplazo
    puntos_muestra = random.sample(puntos_suministro, 1000)
    fabricas_muestra = random.sample(fabricas, 50)

    # Escribir el nuevo archivo
    with open(salida, "w") as file:
        file.write("1000 50 5\n")  # Nueva primera línea
        file.writelines(puntos_muestra)  # Puntos de suministro seleccionados
        file.writelines(fabricas_muestra)  # Fábricas seleccionadas


# Uso del script
procesar_archivo(
    "data_0/kbcl_instances/two_of/A3_7500_150_15.txt",
    "data/positions/p0_1000_50_5.txt",
    semilla=8,
)

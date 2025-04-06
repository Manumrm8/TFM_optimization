import pandas as pd


def delete_dominated_solutions(archive):

    df_solutions = pd.read_csv("./Solutions/" + archive + ".csv")
    df_solutions = df_solutions.reset_index(
        drop=True
    )  # Por si acaso, para mantener los índices ordenados

    i = 0
    while i < len(df_solutions):
        fila_i = df_solutions.loc[i]
        dominada = False
        for j in range(len(df_solutions)):
            if i == j:
                continue
            fila_j = df_solutions.loc[j]
            if (
                (fila_j["f1"] <= fila_i["f1"])
                and (fila_j["f2"] <= fila_i["f2"])
                and (fila_j["f3"] <= fila_i["f3"])
                and (
                    (fila_j["f1"] < fila_i["f1"])
                    or (fila_j["f2"] < fila_i["f2"])
                    or (fila_j["f3"] < fila_i["f3"])
                )
            ):
                df_solutions = df_solutions.drop(index=i).reset_index(drop=True)
                dominada = True
                break
        if not dominada:
            i += 1  # Solo avanzamos si no se ha eliminado la fila actual

    # Resultado final sin dominadas
    print("Soluciones no dominadas:")
    print(df_solutions)

    df_solutions.to_csv("./Solutions/" + archive + ".csv", index=False)
    print("Soluciones guardadas en el archivo original.")
    return

import pandas as pd
import numpy as np


def load_solutions(csv_path, txt_path):
    df_csv_full = pd.read_csv(csv_path)
    df_csv_full.iloc[:, 1] = df_csv_full.iloc[:, 1]
    df_csv = df_csv_full.iloc[:, 1:]  # Solo los objetivos
    df_txt = pd.read_csv(txt_path, sep="\t", header=None)
    df_txt.iloc[:, 0] = df_txt.iloc[:, 0]
    return df_csv_full, df_csv, df_txt


def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def get_pareto_front(solutions):
    is_pareto = np.ones(len(solutions), dtype=bool)
    for i, s in enumerate(solutions):
        for j, t in enumerate(solutions):
            if i != j and dominates(t, s):
                is_pareto[i] = False
                break
    return is_pareto


def classify_solutions(df_obj, pareto_solutions, other_array):
    obj_array = df_obj.values
    colors = []

    for sol in obj_array:
        in_common = any(np.allclose(sol, other, atol=1e-6) for other in other_array)
        in_pareto = any(np.allclose(sol, p, atol=1e-6) for p in pareto_solutions)

        if in_common:
            colors.append("blue")  # En ambos
        elif in_pareto:
            colors.append("green")  # Frente
        else:
            colors.append("red")  # Dominada
    return colors


def get_styled_table(df, colors):
    def color_row(row):
        color = colors[row.name]
        return [f"background-color: {color}"] * len(row)

    return df.style.apply(color_row, axis=1).to_html()

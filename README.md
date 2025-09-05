# A Hybrid Strategic Oscillation with Path Relinking Algorithm for the Multiobjective k-Balanced Center Location Problem

## Data
En la carpeta data se encuentran tanto los archivos de posición, en formato txt, cuya primera fila indica el número de localizaciones a proveer (n), la cantidad de posibles emplazamientos de instalaciones (m) y la cantidad que hay que seleccionar (k) de los m posibles.

En la carpeta distances demand se encuentra la matriz de distancias entre los puntos a suministrar (cada fila) y los posibles emplazamientos (cada columna). Se hizo otra con las distancias entre las posibles instalaciones en la carpeta de distances supply pero no tiene utilidad.

En la carpeta Pareto_front se encuentran los frentes de pareto encontrados.

En la carpeta Pareto_front_paper se encuentran los resultados de un paper que se tomó como referencia

En la carpeta Solutions, se encuentran los resultados al ejecutar tanto el algoritmo GRASP con MAB como sin la parte del reforzado.

## TFM
Aquí se encuentra el archivo latex del trabajo de fin de máster desarrollado.

## tools
En tools se encuentran todas las funciones necesarias para ejecutar los jupyter notebooks.

## Jupyter Notebooks:

- empates_redondeo: El estudio que se hizo para comprobar que en el frente de pareto del paper salen resultados suboptimos al redondear

- extraer_frente_pareto_exacto: Se utilizó para sacar las soluciones exahustivas del WorkSpace 1000_50_5.

- Fine_tuning: Se utilizó para encontrar los mejores parámetros para las variables.

- GRASP_MULTIPROCESSING_MAB: Cuaderno para ejecutar el algoritmo.

- visualization.ipynb: Cuaderno utilizado para visualizar los resultados

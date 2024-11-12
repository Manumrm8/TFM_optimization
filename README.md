# A Hybrid Strategic Oscillation with Path Relinking Algorithm for the Multiobjective k-Balanced Center Location Problem

# Data
Variables:
* n: Cantidad de puntos de demanda, comprendidos en un rectángulo de 1500x1000 
* m: Cantidad de puntos de suministro
* k: Cantidad de puntos de suministro a seleccionar



## kbcl_instances

### pmed
Son las instancias normales

n=1000
m=50
k=5, 10, 15, 20, 25 y 30

pero luego los 3 valores de abajo?
coordenada x, coordenada y y demanda?

cada uno contiene el segundo valor de instancias, entonces tienen que ser todos de facilities? (puntos de suministro?)

40 problemas, número de vertidces/aristas y el número a seleccionar
Después de esa fila tendré todas las aristas. da el final de los vértices y el coste de la arista


### three_of
Instancias más grandes, con n=5000, m=500 y k=50 y 100, las cuales comparamos con 3 funciones objetivo

los largescale son iguales, y las workspace también, el primero indica datos, luego 5000 puntos de demanda y los 500 puntos de suministro

### two_of:
Para comparar "the two objectives two nonuniform"

Las instancias A3 tienen n=7500, m=150 y k=15 ,45 , 75 y 120

Las instancias S1 tienen n=5000, m=100 y k=10, 30, 50 y 80


Son 2 columnas, ¿no existen valores de cantidad a suministrar, o de stock en los puntos de suministro?


## Algoritmo:

Uno trayectorial y otro genético (empezar por este)
Bias random key genético BRKGA con un path relinking implícito.

Después podríamos intentar hacer su adaptación a paralelo (algoritmo en paralelo en python)

Adaptar la construcción de java en python

- Lectura instancias (todo lo necesario que hacer con esta)
- Lectura solución (copiarlas, actualizarlas etc)
- Main con el algoritmo
- Estructuras de construcción
- Estructuras de mejora

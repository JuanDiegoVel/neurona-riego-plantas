import numpy as np

print("Neurona artificial que decide si una planta necesita riego")

# [humedad %, temperatura °C]
X = np.array([
    [80, 18], [70, 22], [65, 28], [55, 25], [50, 32],
    [40, 30], [35, 25], [30, 32], [20, 35], [10, 38]
], dtype=float)

# 1 = regar, 0 = no regar
y = np.array([
    [0], [0], [0], [0], [0],
    [1], [1], [1], [1], [1]
], dtype=float)

print("Entradas \n", X)
print("Respuestas esperadas: \n", y.ravel())

# Se divide por el valor máximo esperado para dejar los datos entre 0 y 1
escala = np.array([100, 50])

X_normalizado = X / escala

print("\nDatos normalizados:")
print(X_normalizado)


def sigmoide(z):
    return 1 / (1 + np.exp(-z))


def entrenar(tasa_aprendizaje, epocas, mostrar_error=False):
    rng = np.random.default_rng(7)
    pesos = rng.normal(size=(2, 1))
    sesgo = 0.0
    errores = []

    for epoca in range(epocas):
        z = X_normalizado @ pesos + sesgo
        probabilidades = sigmoide(z)

        # Error cuadrático medio y sus derivadas
        error = probabilidades - y
        perdida = np.mean(error ** 2)
        errores.append(perdida)
        gradiente_z = 2 * error * probabilidades * (1 - probabilidades) / len(X)
        gradiente_pesos = X_normalizado.T @ gradiente_z
        gradiente_sesgo = np.sum(gradiente_z)

        pesos -= tasa_aprendizaje * gradiente_pesos
        sesgo -= tasa_aprendizaje * gradiente_sesgo

        if mostrar_error and epoca % 1000 == 0:
            print(f"Epoca {epoca:5} | error: {perdida:.4f}")

    return pesos, sesgo, errores


def predecir(datos, pesos, sesgo):
    return sigmoide((datos / escala) @ pesos + sesgo)


def tipo_aprendizaje(errores, error_final):
    if np.any(np.diff(errores) > 0):
        return "inestable"
    if error_final > 0.05:
        return "insuficiente"
    epocas_bajo_005 = np.flatnonzero(np.array(errores) < 0.05)
    if len(epocas_bajo_005) > 0 and epocas_bajo_005[0] < 5000:
        return "rápido"
    return "lento"


tasa_aprendizaje = 0.5
epocas = 10000

print(f"\nEntrenamiento base: tasa {tasa_aprendizaje}, {epocas} épocas")
pesos, sesgo, errores = entrenar(tasa_aprendizaje, epocas, mostrar_error=True)

probabilidades = predecir(X, pesos, sesgo)
respuestas = (probabilidades >= 0.5).astype(int)
error_final = np.mean((probabilidades - y) ** 2)

print("\nPeso humedad:", round(pesos[0, 0], 4))
print("Peso temperatura:", round(pesos[1, 0], 4))
print("Sesgo:", round(sesgo, 4))
print("Error final:", round(error_final, 6))

print("\nProbamos lo aprendido:")
print("Si la probabilidad es >= 0.5 = 1 (regar)")
print("Si es < 0.5 = 0 (no regar)")
for entrada, probabilidad, respuesta in zip(X, probabilidades.ravel(), respuestas.ravel()):
    print(f"{entrada.astype(int)} -> probabilidad {probabilidad:.4f} -> respuesta: {respuesta}")

print("\nSigno de los pesos:")
for nombre, peso in [("humedad", pesos[0, 0]), ("temperatura", pesos[1, 0])]:
    efecto = "baja" if peso < 0 else "sube"
    print(f"Peso {nombre} {peso:+.4f}: si la {nombre} aumenta, la probabilidad de regar {efecto}")

X_nuevos = np.array([[75, 30], [45, 34], [25, 22], [50, 25], [30, 40]], dtype=float)

probabilidades_nuevas = predecir(X_nuevos, pesos, sesgo)
decisiones = (probabilidades_nuevas >= 0.5).astype(int)

print("\nCondiciones nuevas:")
for entrada, probabilidad, decision in zip(X_nuevos, probabilidades_nuevas.ravel(), decisiones.ravel()):
    texto = "regar" if decision == 1 else "no regar"
    print(f"{entrada[0]:.0f}% {entrada[1]:.0f}°C -> probabilidad {probabilidad:.4f} -> {decision} ({texto})")

experimentos = [
    ("Prueba base", 10000, 0.5),
    ("Pocas épocas", 100, 0.5),
    ("Cantidad intermedia", 1000, 0.5),
    ("Más épocas", 20000, 0.5),
    ("Tasa pequeña", 10000, 0.01),
    ("Tasa moderada", 10000, 0.1),
    ("Tasa alta", 10000, 1.0),
    ("Tasa muy alta", 10000, 2.0),
]

print("\nExperimentos:")
print(f"{'Prueba':<20} {'Épocas':>6} {'Tasa':>5} {'Error final':>11} {'Correctas':>9} "
      f"{'P(45%,34°C)':>11}  Aprendizaje")
for nombre, epocas_prueba, tasa_prueba in experimentos:
    pesos_prueba, sesgo_prueba, errores_prueba = entrenar(tasa_prueba, epocas_prueba)
    probabilidades_prueba = predecir(X, pesos_prueba, sesgo_prueba)
    error_prueba = np.mean((probabilidades_prueba - y) ** 2)
    correctas = np.sum((probabilidades_prueba >= 0.5) == y)
    probabilidad_45_34 = predecir(np.array([[45, 34]]), pesos_prueba, sesgo_prueba)[0, 0]
    aprendizaje = tipo_aprendizaje(errores_prueba, error_prueba)
    print(f"{nombre:<20} {epocas_prueba:>6} {tasa_prueba:>5} {error_prueba:>11.6f} "
          f"{correctas:>6}/10 {probabilidad_45_34:>11.4f}  {aprendizaje}")

print("\nComparación de umbrales (sin volver a entrenar):")
for umbral in [0.4, 0.5, 0.6]:
    respuestas_umbral = (probabilidades >= umbral).astype(int)
    correctas = np.sum(respuestas_umbral == y)
    decisiones_umbral = (probabilidades_nuevas >= umbral).astype(int)
    print(f"Umbral {umbral}: entrenamiento {respuestas_umbral.ravel()} ({correctas}/10)"
          f" | nuevos {decisiones_umbral.ravel()}")

assert np.array_equal(respuestas, y.astype(int))

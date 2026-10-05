# Neurona artificial para el riego de plantas

Una neurona con activación sigmoide que decide si una planta necesita riego a partir de
la humedad del suelo y la temperatura. Salida: **1 = regar**, **0 = no regar**.

Los datos son didácticos, no una recomendación agronómica real.

## Variables y datos

- `x1`: humedad del suelo (0 a 100 %)
- `x2`: temperatura ambiental (aprox. 0 a 50 °C)
- `y`: 1 si se debe regar, 0 si no

| Caso | Humedad | Temperatura | Salida |
|:-:|:-:|:-:|:-:|
| 1 | 80 % | 18 °C | 0 |
| 2 | 70 % | 22 °C | 0 |
| 3 | 65 % | 28 °C | 0 |
| 4 | 55 % | 25 °C | 0 |
| 5 | 50 % | 32 °C | 0 |
| 6 | 40 % | 30 °C | 1 |
| 7 | 35 % | 25 °C | 1 |
| 8 | 30 % | 32 °C | 1 |
| 9 | 20 % | 35 °C | 1 |
| 10 | 10 % | 38 °C | 1 |

## Ejecución

```bash
git clone https://github.com/JuanDiegoVel/neurona-riego-plantas.git
cd neurona-riego-plantas
uv sync
uv run main.py
```

La salida completa está en [evidencia/salida_consola.txt](evidencia/salida_consola.txt).

## Cómo funciona

1. Se normalizan las entradas: `X_normalizado = X / escala`, con `escala = [100, 50]`.
2. Suma ponderada: `z = X_normalizado @ pesos + sesgo`.
3. Sigmoide: `probabilidad = 1 / (1 + e^(-z))`.
4. Error cuadrático medio y gradientes: `gradiente_pesos = X_normalizado.T @ gradiente_z`.
5. Se actualizan pesos y sesgo con la tasa de aprendizaje.
6. Salida binaria: `respuesta = (probabilidad >= 0.5).astype(int)`.

Los pesos iniciales salen de `np.random.default_rng(7)`, así todos los experimentos parten
de los mismos valores.

## Resultados de la prueba base (tasa 0.5, 10000 épocas)

| Parámetro | Valor |
|---|---|
| Peso humedad | −19.5032 |
| Peso temperatura | +2.3101 |
| Sesgo | 7.4022 |
| Error final | 0.019645 |
| Correctas | 10/10 |

| Caso | Humedad | Temperatura | Esperado | Probabilidad | Respuesta |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 80 % | 18 °C | 0 | 0.0006 | 0 |
| 2 | 70 % | 22 °C | 0 | 0.0053 | 0 |
| 3 | 65 % | 28 °C | 0 | 0.0183 | 0 |
| 4 | 55 % | 25 °C | 0 | 0.1025 | 0 |
| 5 | 50 % | 32 °C | 0 | 0.2951 | 0 |
| 6 | 40 % | 30 °C | 1 | 0.7285 | 1 |
| 7 | 35 % | 25 °C | 1 | 0.8496 | 1 |
| 8 | 30 % | 32 °C | 1 | 0.9539 | 1 |
| 9 | 20 % | 35 °C | 1 | 0.9941 | 1 |
| 10 | 10 % | 38 °C | 1 | 0.9993 | 1 |

**Signo de los pesos**

- Humedad **negativo**: a más humedad, menos probabilidad de regar.
- Temperatura **positivo**: a más temperatura, más probabilidad de regar.

## Predicciones con condiciones nuevas

Se normalizan con la misma `escala` antes de entrar a la neurona.

| Humedad | Temperatura | Probabilidad | Decisión |
|:-:|:-:|:-:|:-:|
| 75 % | 30 °C | 0.0029 | 0, no regar |
| 45 % | 34 °C | 0.5490 | 1, regar |
| 25 % | 22 °C | 0.9719 | 1, regar |
| 50 % | 25 °C | 0.2325 | 0, no regar |
| 30 % | 40 °C | 0.9677 | 1, regar |

El caso 45 %, 34 °C queda casi en 0.5: está justo en el límite entre las dos clases.

## Experimentos

Criterio del programa: **inestable** si el error sube en alguna época, **insuficiente** si
el error final es mayor a 0.05, **rápido** si baja de 0.05 antes de la época 5000 y
**lento** si lo logra después.

| Prueba | Épocas | Tasa | Error final | Correctas | P(45 %, 34 °C) | Aprendizaje |
|---|--:|--:|--:|:-:|--:|---|
| Prueba base | 10000 | 0.5 | 0.019645 | 10/10 | 0.5490 | rápido |
| Pocas épocas | 100 | 0.5 | 0.165623 | 10/10 | 0.5217 | insuficiente |
| Cantidad intermedia | 1000 | 0.5 | 0.067481 | 10/10 | 0.5793 | insuficiente |
| Más épocas | 20000 | 0.5 | 0.011705 | 10/10 | 0.5263 | rápido |
| Tasa pequeña | 10000 | 0.01 | 0.130048 | 10/10 | 0.5388 | insuficiente (lento) |
| Tasa moderada | 10000 | 0.1 | 0.049728 | 10/10 | 0.5853 | lento |
| Tasa alta | 10000 | 1.0 | 0.011705 | 10/10 | 0.5263 | rápido |
| Tasa muy alta | 10000 | 2.0 | 0.006566 | 10/10 | 0.5082 | rápido |

## Prueba con el umbral

Mismos pesos de la prueba base, sin volver a entrenar.

| Umbral | Entrenamiento | Correctas | Nuevos (75/30, 45/34, 25/22, 50/25, 30/40) |
|:-:|:-:|:-:|:-:|
| 0.4 | 0 0 0 0 0 1 1 1 1 1 | 10/10 | 0 1 1 0 1 |
| 0.5 | 0 0 0 0 0 1 1 1 1 1 | 10/10 | 0 1 1 0 1 |
| 0.6 | 0 0 0 0 0 1 1 1 1 1 | 10/10 | 0 **0** 1 0 1 |

- Solo cambia 45 %, 34 °C (probabilidad 0.549): con 0.6 pasa a "no regar".
- Ningún caso de entrenamiento cambia porque sus probabilidades más cercanas a 0.5 son
  0.295 y 0.729, fuera del rango 0.4 a 0.6.
- Los pesos no cambian porque el umbral se aplica después de la sigmoide y no participa en
  el entrenamiento. Solo mueve el punto de corte.

## Análisis

**1. ¿Por qué fue necesario normalizar?**
La humedad llega a 80 y la temperatura a 38. Sin normalizar, `z` crece demasiado, la
sigmoide se satura en 0 o 1 y el gradiente queda casi en cero. Probando sin normalizar
(tasa 0.5, 10000 épocas), el error se quedó en 0.5 con solo 5/10 correctas. Al dividir
por `[100, 50]` ambas entradas quedan entre 0 y 1 y la misma tasa sirve para los dos pesos.

**2. ¿Dónde se usó `X_normalizado` y para qué se conservó `X`?**
`X_normalizado` se usa en `z = X_normalizado @ pesos + sesgo` y en
`gradiente_pesos = X_normalizado.T @ gradiente_z`. `X` se conserva para mostrar los
resultados en % y °C. Los datos nuevos se dividen por la misma `escala` en `predecir()`.

**3. ¿Qué pasó con 100 épocas?**
Acertó los 10 casos, pero con poca seguridad: error 0.166 y probabilidades entre 0.29 y
0.69, todas cerca de 0.5. Clasificar bien no significa haber aprendido bien.

**4. ¿Más épocas siempre mejoraron mucho?**
No. De 100 a 1000 épocas el error bajó de 0.166 a 0.067, de 1000 a 10000 bajó a 0.020 y
de 10000 a 20000 solo a 0.012. Las correctas fueron 10/10 en todos los casos. Después de
unas miles de épocas la mejora es mínima.

**5. ¿Qué efecto tuvo una tasa muy pequeña?**
Con 0.01 los pasos son 50 veces más cortos. Tras 10000 épocas el error seguía en 0.130,
el mismo resultado que da la tasa 0.5 en solo 200 épocas. Aprendió lento y no alcanzó.

**6. ¿Qué efecto tuvo una tasa alta o muy alta?**
Aprendió más rápido. Con 2.0 el error bajó de 0.05 en la época 494, frente a la 1977 con
0.5, y nunca subió, así que no hubo inestabilidad. Como las entradas están normalizadas y
la derivada de la sigmoide es máximo 0.25, los pasos siguen siendo moderados. La tasa 1.0
con 10000 épocas dio lo mismo que 0.5 con 20000 (error 0.011705).

**7. ¿Qué representa el signo del peso de la humedad?**
Es negativo (−19.50): más humedad significa menos necesidad de riego. Es el peso más
grande porque la humedad sola separa los datos (≥ 50 % no regar, ≤ 40 % regar).

**8. ¿Qué representa el signo del peso de la temperatura?**
Es positivo (+2.31): más calor significa más necesidad de riego por la evaporación. Pesa
unas 8 veces menos que la humedad, así que solo inclina la decisión en casos dudosos como
45 %, 34 °C.

**9. ¿Por qué convertir la probabilidad en 0 o 1 con un umbral?**
La sigmoide da un valor continuo, pero la acción es regar o no regar. El umbral 0.5
convierte esa probabilidad en una decisión. Subirlo o bajarlo hace que riegue con más o
menos facilidad sin reentrenar.

**10. ¿Qué limitaciones tiene esta neurona?**
- Solo usa dos variables: ignora especie, tipo de suelo, lluvia, sol y viento.
- Se entrenó con 10 datos y se comprobó con los mismos.
- Solo puede trazar una frontera recta entre regar y no regar.
- No dice cuánta agua aplicar ni cuándo.
- No considera errores de los sensores.

# Glosario de Data Science — De Cero a la Aplicación

Este documento explica cada concepto técnico del challenge **desde la teoría más básica hasta cómo se usa en el código**. No se asume ningún conocimiento previo de estadística o programación.

---

## Índice

1. [¿Qué es Machine Learning y por qué usarlo?](#1-qué-es-machine-learning-y-por-qué-usarlo)
2. [¿Qué es el clustering? La idea central](#2-qué-es-el-clustering-la-idea-central)
3. [K-Means: el algoritmo paso a paso](#3-k-means-el-algoritmo-paso-a-paso)
4. [El problema de las escalas: por qué hay que normalizar los datos](#4-el-problema-de-las-escalas)
5. [StandardScaler: la solución clásica y sus límites](#5-standardscaler)
6. [RobustScaler: la solución resistente a datos extremos](#6-robustscaler)
7. [log1p: cómo domar una distribución de cola larga](#7-log1p)
8. [Las métricas de evaluación del clustering](#8-métricas-de-evaluación)
9. [Cómo elegir K: utilidad comercial vs. óptimo matemático](#9-cómo-elegir-k)
10. [Por qué se eligieron esas columnas para describir los clusters](#10-por-qué-esas-columnas)
11. [Estrategias comerciales detalladas por segmento](#11-estrategias-comerciales)
12. [El enfoque GenAI híbrido: clustering numérico + LLM](#12-enfoque-genai-híbrido)
13. [dbt incremental: de un script SQL a un pipeline de producción](#13-dbt-incremental)

---

## 1. ¿Qué es Machine Learning y por qué usarlo?

### La idea más simple posible

Machine Learning (ML) es una forma de hacer que una computadora **aprenda patrones en los datos sin que un humano le explique explícitamente qué buscar**.

El enfoque tradicional de programación es: "si el precio es mayor a $10.000 y tiene más de 50 publicaciones, entonces es un vendedor premium". Eso requiere que alguien ya sepa cuáles son las reglas.

Machine Learning invierte esa lógica: **le damos los datos y el algoritmo descubre por sí solo qué patrones existen**. Los humanos interpretamos el resultado.

### ¿Por qué se usa en este challenge?

Mercado Libre tiene 46.524 vendedores con comportamientos completamente distintos. Nadie en el equipo comercial puede revisar manualmente cada uno y decidir a qué grupo pertenece. El algoritmo de clustering analiza 18 características de cada vendedor simultáneamente y los agrupa de forma automática en segmentos con sentido comercial.

### La diferencia entre aprendizaje supervisado y no supervisado

- **Supervisado:** el algoritmo aprende de ejemplos etiquetados. Ejemplo: "estos 1.000 emails son spam, estos 1.000 no lo son — aprendé a distinguirlos". Necesitás tener ejemplos correctos de antemano.
- **No supervisado (clustering):** no hay etiquetas previas. El algoritmo solo recibe los datos y tiene que encontrar grupos por sí solo. Es lo que usamos aquí — nadie le dijo al modelo "este vendedor es premium" antes de entrenarlo.

---

## 2. ¿Qué es el clustering? La idea central

### Analogía con personas en una sala

Imaginá que entrás a una sala con 46.524 personas. No sabés nada de ellas excepto su altura, peso, edad y nivel de ingresos. Se te pide que las agrupes en 5 grupos de forma que cada grupo sea lo más homogéneo posible por dentro, y lo más diferente posible de los otros grupos.

Eso es clustering: **agrupar elementos similares sin saber de antemano cuáles son los grupos correctos**.

### En el contexto del challenge

Cada "persona" es un vendedor de MELI. Sus características son: cuántos ítems tiene, a qué precio vende, qué canal logístico usa, qué reputación tiene, si usa descuentos, etc. El algoritmo agrupa a los vendedores que más se parecen entre sí en esas características.

### ¿Qué es "similitud" para un algoritmo?

Para que una computadora pueda medir si dos vendedores son parecidos, necesita un número. Se usa la **distancia euclidiana**: la misma fórmula del teorema de Pitágoras, pero en muchas dimensiones a la vez.

En dos dimensiones (precio e ítems), la distancia entre el vendedor A ($500, 3 ítems) y el vendedor B ($2.000, 20 ítems) sería:

```
distancia = √((500 - 2000)² + (3 - 20)²)
          = √((-1500)² + (-17)²)
          = √(2.250.000 + 289)
          ≈ 1.500
```

El problema: el precio domina enormemente (1.500 vs. 289). Si el precio varía en miles y los ítems varían en decenas, el algoritmo "ve" solo las diferencias de precio. Por eso hay que **normalizar los datos** antes de aplicar el algoritmo (ver secciones 4, 5 y 6).

---

## 3. K-Means: el algoritmo paso a paso

### ¿Qué significa "K-Means"?

- **K** = el número de grupos que querés encontrar (lo elegís vos antes de correr el algoritmo)
- **Means** = "promedios" en inglés. El algoritmo trabaja calculando el promedio de cada grupo repetidamente.

### Los 4 pasos del algoritmo

**Paso 1: Elegir K puntos de partida aleatorios**

El algoritmo elige K vendedores al azar como "representantes" iniciales de cada grupo. Estos representantes se llaman **centroides**.

**Paso 2: Asignar cada vendedor al centroide más cercano**

Se calcula la distancia entre cada uno de los 46.524 vendedores y los K centroides. Cada vendedor queda asignado al grupo cuyo centroide está más cerca.

**Paso 3: Recalcular los centroides**

Para cada grupo, se calcula el promedio de todas las características de todos los vendedores que quedaron en ese grupo. Ese promedio se convierte en el nuevo centroide.

**Paso 4: Repetir hasta que nada cambie**

Se repiten los pasos 2 y 3 hasta que ningún vendedor cambia de grupo entre una iteración y la siguiente. En ese punto el algoritmo convergió.

### Analogía visual

Imaginá que tirás 5 imanes al azar sobre un mapa donde cada punto es un vendedor. Cada vendedor queda atraído por el imán más cercano (paso 2). Después movés cada imán al "centro de gravedad" de los vendedores que atrajo (paso 3). Los vendedores se reasignan. Repetís esto hasta que los imanes dejan de moverse.

### El rol de `random_state=42` y `n_init=20`

Cada vez que el algoritmo elige los puntos de partida al azar (Paso 1), puede llegar a soluciones distintas. Para resolver esto:
- `random_state=42` fija la semilla de aleatoriedad — si corrés el análisis hoy y en 6 meses, obtenés exactamente los mismos grupos.
- `n_init=20` corre el algoritmo completo 20 veces con distintos puntos de partida y elige la mejor solución entre las 20. Reduce el riesgo de quedar atrapado en una solución subóptima.

---

## 4. El problema de las escalas

### Por qué los datos en bruto no funcionan directamente

Considerá dos variables de un vendedor:
- `avg_price`: varía entre $50 y $50.000 MXN
- `pct_fbm` (proporción de ítems en FBM): varía entre 0 y 1

Cuando el algoritmo calcula la distancia entre dos vendedores, el precio domina completamente la diferencia. La diferencia de $5.000 en precio aplasta cualquier diferencia de 0.5 en pct_fbm. El algoritmo termina agrupando solo por precio y **completamente ignorando la logística, la reputación y los descuentos**.

Para que todas las variables tengan el mismo "peso" en la distancia, hay que **llevarlas a una escala comparable**. Esto se llama **normalización** o **escalado**.

### El caso extremo de este proyecto

El precio máximo del dataset era **$4.770.000.000 MXN** (un error de carga). Si no se trata ese valor antes de escalar, el problema de escala se vuelve catastrófico — ver secciones 5 y 6.

---

## 5. StandardScaler

### ¿Qué hace?

StandardScaler transforma cada variable para que tenga **media = 0** y **desviación estándar = 1**.

Primero necesitamos entender esas dos palabras:

**Media (promedio):** es la suma de todos los valores dividida por la cantidad de valores. Si tenés 5 vendedores con precios $100, $200, $300, $400 y $500, la media es ($100+$200+$300+$400+$500)/5 = $300.

**Desviación estándar:** mide qué tan "esparcidos" están los datos alrededor de la media. Si todos los vendedores tienen el mismo precio, la desviación es 0. Si los precios varían mucho, la desviación es alta.

La fórmula de StandardScaler es:

```
valor_escalado = (valor_original - media) / desviación_estándar
```

Ejemplo concreto para un vendedor con precio = $500, si la media es $1.000 y la desviación estándar es $500:
```
valor_escalado = ($500 - $1.000) / $500 = -1
```

Un vendedor con precio = $1.500:
```
valor_escalado = ($1.500 - $1.000) / $500 = +1
```

Después de escalar, todos los vendedores tienen valores centrados en 0 y la mayoría queda entre -3 y +3. Ahora la distancia entre variables tiene sentido.

### ¿Por qué falló con el outlier de $4,77 mil millones?

El outlier de $4.770.000.000 cambia completamente la media y la desviación estándar:
- Sin outlier: media ≈ $1.000, desvío ≈ $2.000
- Con outlier: media ≈ $954.000.000, desvío ≈ $2.100.000.000

Ahora aplicando la fórmula a un vendedor normal con precio = $500:
```
valor_escalado = ($500 - $954.000.000) / $2.100.000.000 ≈ -0,000454
```

Y a un vendedor con precio = $10.000:
```
valor_escalado = ($10.000 - $954.000.000) / $2.100.000.000 ≈ -0,000454
```

**Ambos vendedores quedan con exactamente el mismo valor escalado**, a pesar de que sus precios son muy distintos ($500 vs $10.000). El algoritmo los ve como idénticos en la dimensión del precio.

El resultado: K-Means agrupa a todos los vendedores normales en un solo blob gigante y deja solo al outlier en su propio cluster. La silueta reportada era 0,62 (aparentemente "excelente"), pero era engañosa — solo medía lo bien separado que estaba el cluster de artefacto del resto.

---

## 6. RobustScaler

### La solución para datos con valores extremos

RobustScaler funciona con el mismo principio que StandardScaler (llevar todo a una escala comparable), pero **usa estadísticas resistentes a los valores extremos** en lugar de la media y el desvío estándar.

En lugar de media y desvío, usa:
- **Mediana:** el valor del "medio" cuando ordenás todos los datos. Si tenés 5 vendedores con precios $100, $200, $300, $400 y $500, la mediana es $300 (el del medio). La mediana no se ve afectada por valores extremos: si el precio máximo fuera $5.000.000 en vez de $500, la mediana seguiría siendo $300.
- **IQR (Rango Intercuartílico):** la diferencia entre el valor en el percentil 75 (el "tercer cuarto" de los datos) y el percentil 25 (el "primer cuarto"). Mide la dispersión ignorando los extremos del 25% inferior y el 25% superior de los datos.

La fórmula de RobustScaler es:

```
valor_escalado = (valor_original - mediana) / IQR
```

Si hay un outlier de $4.770.000.000, **no afecta ni la mediana ni el IQR** porque ese valor está completamente fuera de los percentiles 25 y 75. El escalado queda estable.

### Cuándo usar cada uno

| Situación | Escalador recomendado |
|-----------|----------------------|
| Datos limpios, sin outliers | StandardScaler |
| Datos con outliers extremos que no se pueden eliminar | RobustScaler |
| **Este challenge: log1p ya reduce el efecto del outlier, pero pueden quedar residuales** | **RobustScaler** |

---

## 7. log1p

### ¿Qué es un logaritmo?

Un logaritmo es la respuesta a la pregunta: **¿a qué potencia hay que elevar 10 para obtener este número?**

- log(10) = 1 → porque 10¹ = 10
- log(100) = 2 → porque 10² = 100
- log(1.000) = 3 → porque 10³ = 1.000
- log(1.000.000) = 6 → porque 10⁶ = 1.000.000
- log(4.770.000.000) ≈ 9.7 → porque 10⁹·⁷ ≈ 4,77 mil millones

### ¿Por qué es útil para precios de e-commerce?

Los precios en un marketplace siguen una **distribución de cola larga**: hay miles de vendedores con precios de $100-$1.000, y muy pocos con precios de $50.000+. Si graficás todos los precios en escala normal, la mayor parte de los datos queda aplastada en la izquierda y hay una "cola" larguísima hacia la derecha.

Cuando aplicás el logaritmo, esa cola se comprime:

| Precio original | log1p(precio) |
|-----------------|---------------|
| $10 | 2,4 |
| $100 | 4,6 |
| $1.000 | 6,9 |
| $10.000 | 9,2 |
| $100.000 | 11,5 |
| $4.770.000.000 | 22,3 |

En escala log, la diferencia entre $10 y $100 (2,4 vs 4,6) es comparable a la diferencia entre $1.000 y $10.000 (6,9 vs 9,2). Cada orden de magnitud ocupa el mismo espacio. El outlier de $4,77B sigue siendo el mayor, pero ya no aplasta a todos los demás.

### ¿Por qué `log1p` y no `log`?

`log(0)` es matemáticamente indefinido (menos infinito). Si un vendedor tiene `avg_price = 0`, el cálculo falla. `log1p(x)` significa `log(1 + x)`, así que `log1p(0) = log(1) = 0`. Es una versión segura que funciona cuando los valores pueden ser cero.

---

## 8. Métricas de evaluación

Una vez que el algoritmo termina, necesitamos medir **qué tan buenos son los clusters resultantes**. No existe una sola métrica perfecta — se usan varias juntas.

### Inercia (SSE — Suma de Errores Cuadráticos)

**¿Qué mide?** La suma de las distancias al cuadrado entre cada vendedor y el centroide de su cluster. Es decir: qué tan "apretados" son los grupos.

**Analogía:** si cada cluster fuera un vecindario, la inercia mide la suma de todas las distancias de cada persona a la plaza central de su vecindario. Cuanto menor, más compacto es el vecindario.

**Dirección:** cuanto más baja, mejor (clusters más compactos).

**Limitación crítica:** la inercia **siempre baja** cuando aumentás K. Con K = 46.524 (un cluster por vendedor), la inercia es 0 — cada vendedor es su propio grupo. Pero eso no sirve para nada. Por eso nunca se usa sola.

---

### Silhouette (Coeficiente de Silueta)

**¿Qué mide?** Para cada vendedor, compara dos cosas:
1. **a** = la distancia promedio a todos los demás vendedores del **mismo** cluster (cohesión interna).
2. **b** = la distancia promedio a todos los vendedores del **cluster vecino más cercano** (separación externa).

La fórmula es: `silueta = (b - a) / max(a, b)`

**Cómo interpretar el resultado (va de -1 a +1):**
- **Cerca de +1:** el vendedor está muy cerca de los de su cluster y muy lejos de los de otros clusters. Perfecto.
- **Cerca de 0:** el vendedor está en el borde entre dos clusters — podría estar en cualquiera de los dos.
- **Cerca de -1:** el vendedor estaría mejor en otro cluster. El modelo lo clasificó mal.

**Dirección:** cuanto más alto, mejor.

**¿Por qué 0,22 es aceptable en este proyecto?**

Los vendedores de un marketplace no forman grupos perfectamente separados como si fueran distintas especies animales. Son personas que existen en un continuo — hay infinita variedad de perfiles entre "vendedor pequeño" y "power seller". Una silueta de 0,22 en este contexto es perfectamente razonable y comparable a 0,6+ en datos de laboratorio con grupos claramente delimitados.

Además, la silueta de 0,62 del baseline **no era mejor** — estaba inflada por un cluster de basura: los vendedores con precios erróneos de miles de millones se agrupaban solos y se separaban perfectamente del resto. Era como tener un grupo de "alienígenas" que se diferencia perfectamente de todos los humanos — la silueta sube pero el análisis no sirve para nada.

---

### Davies-Bouldin Index (DBI)

**¿Qué mide?** Para cada par de clusters, calcula qué tan similares son. Combina dos cosas: qué tan disperso es cada cluster (cuánto varían los vendedores dentro de él) y qué tan lejos están los dos clusters entre sí.

**Fórmula conceptual:**
```
DBI = promedio de [ (dispersión del cluster A + dispersión del cluster B) / distancia entre A y B ]
```

**Analogía:** es como medir qué tan bien separados están dos vecindarios. Si el vecindario A es grande (muchas cuadras) y el vecindario B también, y están muy cerca uno del otro, el DBI sube. Si ambos son compactos y están lejos, el DBI baja.

**Dirección:** cuanto más bajo, mejor.
**Valor del proyecto:** 1,2658 — clusters razonablemente separados.

---

### Calinski-Harabász Index (CH)

**¿Qué mide?** El ratio entre la varianza *entre* clusters y la varianza *dentro* de los clusters.

En términos simples: **¿cuánto más distintos son los grupos entre sí comparado con cuán parecidos son los miembros dentro de cada grupo?**

**Analogía:** imaginá 5 países. Si las personas dentro de cada país son muy similares entre sí y los países son muy diferentes unos de otros, el CH es alto. Si hay mucha variedad dentro de cada país y poco entre países, el CH es bajo.

**Dirección:** cuanto más alto, mejor.
**Valor del proyecto:** 20.879 — alta separación entre grupos.

---

### cluster_max_pct (métrica de utilidad comercial)

**¿Qué mide?** El porcentaje de vendedores que cayó en el cluster **más grande**.

Esta métrica **no existe en las librerías de ML** — la calculamos nosotros porque las métricas matemáticas no miden si el resultado es *útil para el negocio*.

**¿Por qué importa?** Si el 99% de los vendedores está en un solo cluster, el modelo no segmentó nada — es como si hubieras dividido todos los vendedores en dos grupos: "todos" y "nadie". No sirve para estrategias focalizadas.

Un clustering accionable tiene el cluster mayor por debajo del ~50%.

---

## 9. Cómo elegir K

### La tabla completa con interpretación

| K | Silhouette | DBI | CH | Cluster mayor | ¿Es útil? |
|---|-----------|-----|----|---------------|-----------|
| 2 | 0,524 | 0,953 | 27.383 | **84,6%** | ❌ El 84,6% está en un grupo. No segmenta. |
| 3 | 0,484 | 1,033 | 25.506 | 77,6% | ❌ Todavía 3 de cada 4 sellers en un grupo. |
| 4 | 0,330 | 1,198 | 23.000 | 62,6% | ⚠️ Mejora, pero sigue dominado por un grupo gigante. |
| **5** | **0,219** | **1,266** | **20.879** | **41,3%** | ✅ Ningún grupo domina. Aparecen los drop-shippers. |
| 6 | 0,220 | 1,228 | 20.262 | 41,1% | ⚠️ Silhouette casi idéntica. Clusters se fragmentan sin ganar insight. |

**Lo que cuenta la tabla:**

- De K=2 a K=4: las métricas matemáticas son mejores (silueta más alta, DBI más bajo, CH más alto), **pero el cluster mayor es enorme**. El 84,6% de los vendedores en un solo grupo significa que el modelo prácticamente no segmentó.

- **K=5 es el punto de inflexión** donde el cluster mayor cae a 41,3%. Además, con K=5 y `pct_ds` como feature, el algoritmo separa por primera vez a los drop-shippers (canal DS) como un segmento propio — algo que con K=4 no ocurría.

- De K=6 en adelante: la silueta casi no mejora (0,220 vs 0,219) y los grupos empiezan a dividirse en segmentos tan pequeños que pierden utilidad comercial.

**Conclusión:** la elección de K=5 es una **decisión de utilidad comercial** consciente y defendible, no simplemente seguir la métrica que da el número más alto.

---

## 10. Por qué esas columnas

La tabla de perfil muestra estas columnas para describir cada cluster. No son las variables del modelo (esas son las 18 features log-transformadas). Son las variables **más interpretables para el negocio**:

| Columna | Qué representa en la vida real | Por qué la incluimos |
|---------|-------------------------------|----------------------|
| `median_price` | El precio del "vendedor típico" del segmento — la mitad de sus ítems están por debajo de este precio y la mitad por arriba | Es el mejor indicador del "ticket" del vendedor. La mediana es más robusta que el promedio cuando hay ítems con precios muy distintos dentro del mismo seller. |
| `items_count` | Cuántas publicaciones activas tiene el vendedor en promedio | Distingue los vendedores de entrada (1-2 ítems) de los power sellers (40+ ítems). Es el indicador más claro del tamaño de la operación. |
| `pct_fbm` | De cada 100 ítems del vendedor, cuántos están en Fulfillment by Mercado Libre | FBM significa que el vendedor depositó stock en los centros de distribución de MELI. Alta integración con la plataforma. |
| `pct_xd` | % de ítems en Cross-Docking | El canal dominante en México (63%). El vendedor entrega el producto a MELI solo cuando hay una venta. |
| `pct_ds` | % de ítems en Drop Shipping | El vendedor no tiene stock propio — envía directamente desde el proveedor. Más riesgo de cancelaciones. |
| `seller_reputation_score` | La reputación numérica (1=rojo/penalizado, 9=platino) | Es la "nota" que le ponen los compradores al vendedor históricamente. |
| `pct_items_with_discount` | De cada 100 publicaciones, cuántas tienen un precio de oferta activo | Indica la estrategia de pricing: algunos segmentos tienen el 92% de sus ítems con descuento activo. |
| `sellers` | Cantidad de vendedores en el cluster | Para dimensionar el segmento. |
| `pct` | % del total de vendedores que pertenece a este cluster | Para entender el peso relativo de cada segmento. |

---

## 11. Estrategias comerciales por segmento

### C2 · FBM Discount Players (11,7% · 5.426 sellers)

**¿Quiénes son?**

Vendedores que descubrieron el "truco" del algoritmo de MELI: si publicás un ítem a un precio "original" más alto y después lo "descontás" al precio real, el algoritmo de búsqueda te da más visibilidad que si simplemente publicás al precio real sin descuento. El 92% de sus ítems tiene un descuento activo.

Combinan esto con FBM (sus ítems están en los depósitos de MELI), lo que les da envío rápido y mayor confianza del comprador.

**¿Por qué son el cluster más compacto (silueta 0,408)?**

Porque la combinación FBM + descuentos agresivos es muy específica y muy diferente a cualquier otro perfil. Son fácilmente identificables.

**Estrategias:**

1. **MercadoLíder Acelerado:** ya tienen el perfil para ser MercadoLíderes. Darles acceso anticipado al programa (mayor visibilidad, soporte prioritario, comisiones reducidas) antes de que lo pidan genera lealtad y reduce el riesgo de que migren a otra plataforma.

2. **Alerta automática de margen:** monitorear si el descuento promedio supera el 40%. Un vendedor que aplica 40%+ de descuento puede estar vendiendo con pérdida. La alerta dispara un contacto: "Detectamos que tus descuentos podrían estar comprimiendo tu margen — te mostramos cómo optimizar sin perder visibilidad."

3. **Herramienta de pricing dinámico:** ofrecerles una herramienta que ajusta automáticamente el precio según la competencia. Les ahorra el trabajo manual de ajustar precios y optimiza el margen sin perder posición en los resultados de búsqueda.

**KPIs a monitorear:** tasa de retención en FBM mensual; descuento promedio (alerta si >40%); GMV mensual por seller (alerta si cae >20%).

---

### C3 · Power Sellers Multi-Categoría (1,8% · 846 sellers)

**¿Quiénes son?**

Los vendedores más consolidados del marketplace. Con 42 ítems en promedio y presencia en múltiples categorías, representan una operación madura. Son solo el 1,8% de los sellers pero probablemente generan el 15-25% del GMV total.

Internamente son dos arquetipos en un mismo cluster:
- Algunos tienen stock propio masivo y usan XD.
- Otros son drop-shippers de alto volumen (DS = 37%) — sin stock propio pero con alta rotación.

Los une el tamaño del catálogo y la diversidad de categorías, por eso el algoritmo los agrupa.

**Estrategias:**

1. **Account Manager dedicado:** es el único segmento donde el costo de un AM tiene sentido por el GMV que representa. Reunión mensual de revisión: cómo va el catálogo, oportunidades de FBM, benchmarks vs competidores.

2. **Migración DS → FBM en los top-ítems:** identificar los 5 ítems de mayor rotación de cada seller DS-heavy y proponerles una prueba piloto de FBM. El argumento con datos: FBM reduce las cancelaciones en un 35-40% en MELI, lo que protege la reputación del seller y la experiencia del comprador.

3. **Early adopters de nuevas funcionalidades:** estos sellers son los que más usan la plataforma, los que más la entienden y los que pueden dar el mejor feedback. Incluirlos en betas genera goodwill y mejores productos.

**KPIs:** tasa de cancelación por seller; ratio FBM/DS (objetivo: aumentar FBM); GMV mensual (alerta si cae >10%).

---

### C4 · Alto Ticket Sin Historial (29,1% · 13.540 sellers)

**¿Quiénes son?**

Vendedores recién llegados a la plataforma (o muy poco activos) con una única publicación de un producto de alto valor. El precio mediano es ~$12.151 MXN — los artículos pueden ser: laptops, celulares de gama alta, cámaras profesionales, equipamiento médico, vehículos, instrumentos musicales.

La reputación baja (4,36/9) **no indica mal desempeño** — simplemente no han acumulado suficiente historial de ventas como para que los compradores los califiquen.

**El problema clave:** si este tipo de vendedor no concreta su primera venta en los primeros 30-60 días, la probabilidad de que abandone la plataforma es muy alta.

**Estrategias:**

1. **Programa Primera Venta (trigger automático):**
   - Día 7 sin ventas → email con checklist de optimización: "¿Tus fotos muestran el producto desde todos los ángulos? ¿Tu descripción responde las preguntas más frecuentes? ¿Tu precio está dentro del rango de la categoría?"
   - Día 14 sin ventas → oferta de crédito en Product Ads ($500-$1.000 MXN) para dar visibilidad al ítem.
   - Día 30 sin ventas → llamada de activación o encuesta de abandono.

2. **Badge "Vendedor Verificado":** un badge visible en la publicación que dice que la identidad y el negocio del seller fueron verificados por MELI. Para un comprador que va a gastar $12.000 en un artículo, ver ese badge puede ser la diferencia entre comprar o no comprar.

3. **Precio sugerido:** mostrar en el panel del seller el rango de precios de su categoría ($X mínimo, $Y promedio, $Z máximo). Si su precio está un 50% por encima del promedio de la categoría, puede que simplemente no lo sepan.

**Churn risk alto:** sin primera venta, sin red, sin historial — alta probabilidad de abandono temprano.

---

### C0 · Vendedores Activos Multi-Item (16,1% · 7.487 sellers)

**¿Quiénes son?**

Vendedores que ya superaron la etapa inicial: tienen entre 7 y 12 ítems en promedio y una reputación alta (7,37/9 — la más alta del modelo en promedio). Están *en transición* hacia convertirse en Power Sellers.

Son el cluster con la silueta más baja (0,109) porque están en el "medio" del espectro — no son ni básicos ni power sellers. Eso es exactamente lo que esperaríamos: un segmento que está cambiando tiene fronteras difusas.

**Estrategias:**

1. **Incentivo de adopción FBM:** "El 87% de los sellers con tu perfil que migraron a FBM vieron su tasa de conversión aumentar un 23% en 60 días." Oferta: primeras 30 unidades con logística FBM subsidiada (periodo de prueba). El objetivo es que experimenten el beneficio antes de comprometerse.

2. **Challenge de catálogo:** "Agregá 3 ítems más este mes y desbloqueá acceso a Product Ads con $500 MXN de crédito." La gamificación (desafíos con recompensas) es efectiva para llevarlos del rango de 7-12 ítems al rango de 15+ donde empiezan a comportarse como Power Sellers.

3. **Benchmarks personalizados:** mostrar en el panel cómo se comparan sus métricas (precio, descuento, stock) contra la mediana de sellers del mismo segmento en su categoría. La información contextual genera acciones sin necesidad de contacto directo.

**Churn risk bajo-medio:** ya tienen inversión en la plataforma (tiempo, catálogo, reputación), pero si no crecen en 6 meses pueden estabilizarse como "vendedores dormidos".

---

### C1 · Masa Básica — Primera Publicación (41,3% · 19.225 sellers)

**¿Quiénes son?**

El "long tail" del marketplace — el término "long tail" describe el fenómeno donde la mayoría de los participantes son pequeños. Aquí: 1-2 ítems publicados, sin estrategia de precio visible, sin descuentos, canal XD estándar.

Dentro de este grupo hay perfiles muy distintos:
- Vendedores ocasionales que publicaron algo una vez y nunca más volverán.
- Futuros C4 (que van a agregar ítems de alto valor).
- Futuros C0 (que van expandiendo el catálogo progresivamente).

**La advertencia de negocio:** a escala de plataforma, este grupo tiene el menor retorno por atención directa. Un Account Manager dedicado a este segmento no es rentable. La estrategia es **automatización masiva**.

**Estrategias:**

1. **Sub-segmentación por señal de actividad** — dentro de C1, separar:
   - ✅ Sellers con al menos 1 venta en 30 días → programa de crecimiento acelerado hacia C0.
   - ⚠️ Sellers con 0 ventas en 30 días → campaña de reactivación.
   - ❌ Sellers con 60+ días sin ventas y sin respuesta a mensajes → reducción de visibilidad (no eliminación).

2. **Educación automatizada:** secuencia de 5 emails en el primer mes, sin intervención humana:
   - Email 1: "Cómo tomar fotos que venden"
   - Email 2: "El precio justo para tu categoría"
   - Email 3: "Por qué los descuentos mejoran tu visibilidad"
   - Email 4: "Cómo responder rápido a los mensajes y subir tu reputación"
   - Email 5: "Tu primer mes en MELI: qué hacer si todavía no vendiste"

3. **Umbral de calidad mínima:** sellers con más de 60 días sin ventas y tasa de respuesta a mensajes < 50% → reducir visibilidad en los resultados de búsqueda. No eliminarlos, pero priorizar los sellers activos para la experiencia del comprador.

**KPIs:** tasa de conversión a C0 en 90 días (objetivo: 5-8%); % con al menos 1 venta mensual; engagement con materiales educativos (tasa de apertura de emails).

---

## 12. Enfoque GenAI híbrido

### El problema que resuelve

Después de que el algoritmo de K-Means termina, tenemos una tabla con los promedios de las 18 variables para cada cluster. Esa tabla la puede leer un Data Scientist. Pero el equipo comercial de MELI necesita una descripción en lenguaje natural: "estos son vendedores recién llegados con productos premium que necesitan apoyo en su primera venta", no "este cluster tiene log_median_price = 9.4 y seller_reputation_score = 4.36".

El LLM (Large Language Model, como Claude) actúa como **traductor** entre los números del modelo y el lenguaje de negocio.

### ¿Por qué se descartó clusterizar sobre embeddings de texto?

La primera sugerencia de los modelos de IA fue: "usá embeddings de los títulos de las publicaciones para clusterizar". Fue descartada.

**¿Qué es un embedding?** Es una representación numérica de un texto que captura su *significado semántico*. Dos textos con significado similar tienen embeddings similares (y por ende estarían cerca en el espacio de clustering).

**El problema:** si clusterizás por similitud de títulos, obtenés clusters de **categorías de productos**:
- Cluster "Tecnología": iPhone, MacBook, Samsung Galaxy
- Cluster "Moda": zapatillas, remeras, jeans
- Cluster "Hogar": sillas, mesas, estantes

Pero dos vendedores pueden vender exactamente el mismo iPhone y ser perfiles de negocio completamente opuestos:

| Característica | Vendedor Premium | Drop-shipper |
|---------------|------------------|--------------|
| Qué vende | iPhone 15 Pro | iPhone 15 Pro |
| Reputación | Verde Platino (9/9) | Rojo (1/9) |
| Stock | 200 unidades | 1 unidad |
| Logística | FBM 100% | DS 100% |
| Precio | $18.000 MXN | $17.500 MXN |
| Cancelaciones | <1% | 25% |

Con embeddings, quedarían en el **mismo cluster** porque venden el mismo producto. Con el clustering numérico sobre features de comportamiento, quedan en **clusters opuestos**.

**El principio:** los textos de los títulos describen *qué* se vende. Las features numéricas describen *cómo* opera el negocio. La pregunta del challenge es sobre el *cómo*.

### El flujo híbrido paso a paso

```
1. K-Means numérico
   → 46.524 sellers × 18 features numéricas
   → 5 clusters con centroides

2. summarize_clusters()
   → Una fila por cluster con los promedios de todas las variables
   → Ejemplo: Cluster 2: pct_fbm=0.69, avg_discount=0.92, reputation=7.70

3. Serializar a JSON
   → Formato estructurado que el LLM puede leer
   → [{"cluster_id": 2, "avg_pct_fbm": 0.69, ...}]

4. Armar el prompt para el LLM
   → "Sos un analista comercial de MELI. Describí cada cluster en lenguaje natural
      y proponé 2 acciones comerciales concretas."

5. LLM responde
   → "Cluster 2: vendedores altamente integrados con FBM que usan descuentos
      agresivos para ganar visibilidad en el feed de búsqueda..."
```

**¿Por qué es superior a solo mirar los números?**

Un analista de negocio no puede interpretar una tabla con 18 columnas numéricas de forma intuitiva. El LLM convierte esos datos en una narrativa accionable. Los clusters están anclados en datos reales (no en intuición del modelo), y el LLM aporta el lenguaje de negocio. Juntos son más poderosos que cualquiera de los dos por separado.

---

## 13. dbt incremental

### El problema de escala que resuelve

En el análisis local, procesamos 185.250 ítems de un solo día con pandas en segundos. En producción, la misma tabla va a crecer así:

```
Hoy:          185.250 ítems  (1 día)
En 30 días: 5.557.500 ítems  (~30 días de snapshots)
En 1 año:  67.616.250 ítems  (~365 días)
En 5 años: 338.081.250 ítems (~1.825 días)
```

Pandas en una laptop no puede procesar esos volúmenes. La solución es BigQuery (la base de datos en la nube de Google).

### ¿Qué es dbt?

dbt (data build tool) es una herramienta que permite escribir transformaciones de datos en SQL con **versionado, documentación y tests automáticos**. Es el estándar de la industria para pipelines analíticos en la nube.

La diferencia con ejecutar un script SQL manualmente:

| Sin dbt | Con dbt |
|---------|---------|
| Ejecutás el SQL cuando te acordás | Se ejecuta automáticamente según un schedule |
| Si algo falla, te enterás cuando alguien se queja | Los tests de calidad fallan antes de que los datos lleguen al dashboard |
| No hay documentación de qué tabla genera qué | Lineage automático: sabés exactamente qué tabla alimenta a qué |
| Para actualizar la tabla, reprocesás todo | Modo incremental: solo procesás los datos nuevos |

### ¿Qué es el modo incremental?

Sin incrementalidad, cada vez que corre el pipeline:
1. Borra toda la tabla de features de sellers.
2. Lee **todos** los 2,5 TB de datos históricos.
3. Recalcula las features para **todos** los sellers.
4. Escribe la tabla nueva.

Costo: 2,5 TB × $5/TB = **$12,50 por corrida**. Si corre 50 veces al día: **$625/día ≈ $227.500/año**.

Con incrementalidad (`insert_overwrite`):
1. Lee solo el snapshot del día nuevo (6,85 GB).
2. Calcula las features solo para los sellers del nuevo snapshot.
3. Sobreescribe únicamente la partición del mes actual en BigQuery.

Costo: 6,85 GB × $0,25/GB = **$1,70 por día ≈ $620/año**.

**Ahorro: ~$226.880/año.**

### ¿Cómo funciona `is_incremental()` en el código?

```sql
{% if is_incremental() %}
  WHERE tim_day = DATE('{{ var("snapshot_date") }}')
{% endif %}
```

Esta línea de código es la clave:
- **Primera vez que corre:** `is_incremental()` es `False` → no hay `WHERE` → procesa toda la historia.
- **Corridas siguientes:** `is_incremental()` es `True` → el `WHERE` filtra solo el día nuevo → solo lee 6,85 GB.

### ¿Está implementado en el código del repo?

**Parcialmente, de forma ilustrativa.** `bigquery_strategy.py` contiene:
- El DDL de la tabla raw en BigQuery (con partición mensual y clustering físico).
- El SQL de la tabla de features agregadas por seller.

Lo que **no está en el repo** (requeriría conexión a BigQuery y credenciales reales de MELI):
- Un proyecto dbt real (`dbt_project.yml`, carpeta `models/`, `profiles.yml`).
- La conexión autenticada a BigQuery.
- Los modelos de staging (limpieza de datos antes de agregar).

Para el challenge, el código local demuestra que el análisis es correcto y reproducible. La arquitectura cloud se defiende conceptualmente con el SQL ilustrativo. Una implementación productiva real crearía el proyecto dbt completo sobre la infraestructura de MELI.

---

*Glosario del Challenge Data Analytics Engineer — Mercado Libre. 2026-06-17.*

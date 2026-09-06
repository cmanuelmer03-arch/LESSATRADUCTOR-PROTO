# Traductor de Lengua de Señas Salvadoreña (LESSA)

Componente de reconocimiento de lengua de señas del proyecto **"Asistente
de discapacidad: Traductor y Detector de obstáculos para personas
discapacitadas"**. Usa la cámara web para detectar la mano (MediaPipe),
extraer sus características geométricas y clasificarlas con un modelo de
Machine Learning (scikit-learn) entrenado por ustedes mismos con sus
propias señas.

---

## 1. Estructura del proyecto

```
lessa_traductor/
├── main.py                  # Aplicación con interfaz gráfica (la que se ejecuta día a día)
├── utils.py                 # Detección de manos (MediaPipe) + extracción de características
├── captura_datos.py         # Herramienta para grabar datos de entrenamiento con tu cámara
├── entrenamiento.py         # Entrena el modelo con los datos capturados
├── modelo.py                # Carga el modelo entrenado y hace predicciones
├── generar_datos_demo.py    # (Opcional) genera datos sintéticos para probar el pipeline
├── requirements.txt         # Librerías necesarias
├── data/
│   └── datos_senas.csv      # Se crea al usar captura_datos.py (sus datos reales)
└── modelos/
    └── modelo_lessa.pkl     # Se crea al usar entrenamiento.py (el modelo ya entrenado)
```

### ¿Qué hace cada archivo?

| Archivo | Función |
|---|---|
| `utils.py` | Detecta la mano con MediaPipe y convierte sus 21 puntos en un vector numérico (coordenadas normalizadas + ángulos de los dedos + distancias entre dedos). Es el "corazón" técnico del reconocimiento. |
| `captura_datos.py` | Script de consola para grabar, con su propia cámara, ejemplos de cada seña y guardarlos en un CSV etiquetado. |
| `entrenamiento.py` | Lee ese CSV y entrena un clasificador `RandomForest`. Guarda el modelo en `modelos/modelo_lessa.pkl`. |
| `modelo.py` | Envoltorio simple para cargar ese modelo y pedirle predicciones desde `main.py`. |
| `main.py` | La aplicación final: ventana con video en vivo, botones Iniciar/Detener, texto reconocido en pantalla. |
| `generar_datos_demo.py` | Genera un CSV con datos **inventados matemáticamente** (no de una mano real) solo para comprobar que el entrenamiento funciona antes de ir a grabar datos de verdad. |

---

## 2. ¿Existen datos de LESSA ya listos, o hay que entrenar?

Investigamos esto para ustedes. Un par de datos importantes:

- El diccionario oficial de LESSA (impulsado por la Asociación Salvadoreña de Sordos) agrupa alrededor de 500 palabras y sus señas correspondientes, organizadas en clasificaciones temáticas como geografía, partes del cuerpo o familia. La UES también publicó un diccionario especializado (DEES-LESSA) enfocado en educación superior, con cerca de 600 términos. Ninguno de los dos viene en formato "landmarks + etiqueta" listo para un modelo de IA; son diccionarios visuales para personas, no datasets de entrenamiento.
- Encontramos un dataset público llamado *"Lenguaje de Señas de El Salvador (LESSA)"* en Kaggle, orientado a clasificación de imágenes. No pudimos confirmar su contenido exacto (cuántas señas, si son estáticas o dinámicas, con cuántas variaciones). Si quieren explorarlo como punto de partida, revísenlo en Kaggle buscando ese nombre, pero probablemente de todas formas necesiten complementar con sus propias grabaciones.
- **No existe** (que hayamos podido encontrar) un dataset abierto de LESSA con 150 palabras ya convertidas a landmarks de MediaPipe, listo para entrenar sin grabar nada.

**Conclusión práctica:** el sistema está diseñado para que *ustedes* generen ese dataset fácilmente con `captura_datos.py`, usando el diccionario visual de la UES como guía de referencia de cada seña. Esto además es un punto fuerte para la parte de metodología de su proyecto de investigación: ustedes mismos construyen el dataset, lo documentan y pueden justificar cada decisión.

---

## 3. Instalación paso a paso en PyCharm

> ¿Van a usar **Visual Studio Code** en vez de PyCharm? Sigan la guía
> completa en [`INSTALACION_VSCODE.md`](./INSTALACION_VSCODE.md) —
> mismo resultado, con extensiones y comandos específicos para VS Code.

### 3.1 Requisitos previos
- **Python 3.10, 3.11 o 3.12** (recomendado; MediaPipe no siempre soporta de inmediato la versión de Python más reciente). Revisen su versión con `python --version`.
- PyCharm instalado (Community o Professional, cualquiera funciona).
- Una cámara web funcionando.

### 3.2 Crear el proyecto en PyCharm
1. Abran PyCharm → **File → Open...** y seleccionen la carpeta `lessa_traductor` (o **New Project** si prefieren empezar vacío y copiar estos archivos adentro).
2. Cuando PyCharm pregunte por el intérprete, elijan **New environment using Virtualenv** (esto crea un entorno virtual aislado solo para este proyecto, evitando conflictos con otras cosas instaladas en su computadora). Confirmen que la versión base sea Python 3.10–3.12.
3. Esperen a que PyCharm termine de indexar el proyecto.

### 3.3 Instalar las librerías necesarias
**Opción A — Terminal integrada de PyCharm (recomendada, la más simple):**
1. Abajo en PyCharm hagan clic en la pestaña **Terminal** (se abre ya con el entorno virtual del proyecto activado, verán `(venv)` al inicio de la línea).
2. Ejecuten:
   ```bash
   pip install -r requirements.txt
   ```
3. Esperen a que termine (MediaPipe y OpenCV pesan varios cientos de MB, puede tardar unos minutos).

**Opción B — Interfaz gráfica de PyCharm:**
1. **File → Settings** (en Mac: **PyCharm → Preferences**) → **Project: lessa_traductor → Python Interpreter**.
2. Clic en el botón **+** (Add Package).
3. Busquen e instalen, uno por uno: `opencv-python`, `mediapipe`, `numpy`, `pandas`, `scikit-learn`, `joblib`, `pillow`.

### 3.4 Verificar la instalación
En la terminal integrada:
```bash
python -c "import cv2, mediapipe, numpy, pandas, sklearn, joblib, PIL; print('Todo instalado correctamente')"
```
Si no aparece ningún error, están listos para continuar.

> **Nota sobre la cámara:** en Windows, si `cv2.VideoCapture(0)` no encuentra la cámara, prueben `cv2.VideoCapture(1)` (cambiar el `0` por `1` dentro de `captura_datos.py` y `main.py`) — a veces el índice cambia según cuántos dispositivos de video tenga la laptop.

---

## 4. Flujo de trabajo completo

### Paso 1 — (Opcional) Probar que todo funciona con datos de ejemplo
Antes de grabar señas reales, pueden validar que el pipeline entero corre sin errores:
```bash
python generar_datos_demo.py
python entrenamiento.py
```
Esto crea un modelo "de juguete" con 5 señas ficticias (datos inventados matemáticamente, no de una mano real) solo para confirmar que `entrenamiento.py` y `modelo.py` funcionan en su computadora antes de invertir tiempo grabando señas de verdad. Pueden borrar después `data/datos_senas.csv` y `modelos/modelo_lessa.pkl` para empezar limpio.

### Paso 2 — Grabar sus propias señas (los datos reales)
```bash
python captura_datos.py
```
1. Escriban el nombre de la primera seña (ej. `hola`), siguiendo el diccionario visual de LESSA como referencia de cómo se hace correctamente.
2. Coloquen la mano frente a la cámara haciendo la seña.
3. Presionen **`c`** para capturar cada muestra. Repitan 30-50 veces por seña, variando levemente:
   - el ángulo de la mano,
   - la distancia a la cámara,
   - la posición dentro del encuadre.
4. Presionen **`n`** para pasar a la siguiente seña (pide un nuevo nombre).
5. Presionen **`q`** cuando terminen. Todo queda guardado en `data/datos_senas.csv`.

**Recomendaciones para datos de mejor calidad:**
- Graben en al menos 2-3 condiciones de luz distintas (luz natural de ventana, luz artificial de foco, un ambiente algo más oscuro). Esto evita que el modelo "aprenda" la iluminación del cuarto en vez de la forma de la mano.
- Si el proyecto lo van a usar varias personas, que cada una grabe también sus propias muestras — manos de distinto tamaño ayudan a que el modelo generalice.
- Empiecen con 5-10 señas cotidianas (hola, gracias, por favor, bien, adiós, sí, no, ayuda, baño, te quiero) antes de escalar a un vocabulario más grande; es más fácil depurar errores con pocas clases.

### Paso 3 — Entrenar el modelo con sus datos reales
```bash
python entrenamiento.py
```
Verán en la terminal cuántas muestras hay por seña y qué tan bien predice el modelo en un conjunto de prueba (accuracy y reporte por clase). Si alguna seña sale con baja precisión, casi siempre significa que necesita más muestras o que se parece demasiado a otra seña ya registrada.

### Paso 4 — Ejecutar la aplicación
```bash
python main.py
```
Se abre la ventana: clic en **"Iniciar cámara"**, hagan las señas frente a la cámara y el texto reconocido va apareciendo en el cuadro de la derecha. **"Detener cámara"** la apaga, y **"Limpiar texto"** vacía el cuadro de texto acumulado.

---

## 5. Cómo funciona el reconocimiento (para su documentación técnica)

1. **Captura de video:** OpenCV lee cada frame de la cámara (`cv2.VideoCapture`).
2. **Detección de mano:** MediaPipe (`HandLandmarker`, la API vigente de Google) localiza la mano en el frame y devuelve 21 puntos clave (nudillos, puntas de dedos, muñeca) en coordenadas normalizadas (x, y, z).
3. **Extracción de características** (`utils.extraer_caracteristicas`): esos 21 puntos crudos dependen de dónde está la mano en pantalla, lo cual *no* nos interesa para reconocer la seña. Por eso los transformamos en un vector invariante a posición y escala, combinando:
   - Coordenadas normalizadas (centradas en la muñeca, escaladas por el tamaño de la mano).
   - Ángulos de flexión de cada uno de los 5 dedos (¿estirado o doblado?).
   - Distancias entre puntas de dedos y la muñeca, y entre dedos vecinos (¿juntos o separados?).
4. **Clasificación:** ese vector se le pasa a un `RandomForestClassifier` de scikit-learn, entrenado previamente con sus propios ejemplos, que devuelve la seña más probable y su nivel de confianza.
5. **Estabilización:** para evitar que el texto "parpadee" con predicciones erróneas de un solo frame, `main.py` solo confirma una seña cuando se repite de forma consistente durante varios frames seguidos (ver `TAMAÑO_BUFFER_ESTABILIZACION` en `main.py`).

### ¿Por qué RandomForest y no una red neuronal con TensorFlow?
Con la cantidad de datos que normalmente se recolecta a mano (decenas o cientos de muestras por seña, no miles), un RandomForest generaliza mejor, entrena en segundos sin GPU, y es más fácil de depurar para un proyecto de bachillerato/universidad. Si más adelante recolectan un dataset mucho más grande, pueden migrar a una red neuronal (Keras/TensorFlow) reutilizando exactamente el mismo vector de características de `utils.py` como entrada.

---

## 6. Cómo agregar nuevas señas / escalar hacia las 150 palabras

El sistema no tiene un límite fijo de señas: cada seña nueva es simplemente una etiqueta más en el CSV.

1. Corran `python captura_datos.py` de nuevo, presionen `n` y escriban el nombre de la nueva seña (o agréguenla desde el inicio si es la primera vez). Los datos se van **añadiendo** al mismo `data/datos_senas.csv`, no lo borra.
2. Vuelvan a correr `python entrenamiento.py`. El modelo se re-entrena desde cero con todas las señas acumuladas hasta ese momento.
3. No necesitan tocar `main.py` ni `modelo.py`: automáticamente reconocen cualquier etiqueta que exista en el modelo entrenado.

**Para escalar a ~150 palabras**, la recomendación es hacerlo por lotes:
- Agrupen las palabras por categorías (saludos, familia, números, lugares…), como ya lo hace el propio diccionario de LESSA.
- Graben y entrenen por lotes de 15-20 señas nuevas a la vez, revisando el `classification_report` de `entrenamiento.py` después de cada lote — así detectan rápido si dos señas nuevas se confunden entre sí (por ejemplo, si tienen formas de mano muy parecidas) antes de seguir agregando más.
- Recuerden que este proyecto solo cubre **señas estáticas de una mano**, tal como definieron el alcance. Señas con movimiento o de dos manos necesitarían capturar una secuencia de frames en vez de una sola postura, lo cual es un cambio de diseño más grande.

---

## 7. Pruebas y validación sugeridas

- **Condiciones de luz:** prueben la app en luz natural de día, luz artificial de noche, y contraluz (ventana detrás de la persona). Anoten en qué condición baja la precisión — es información valiosa para la sección de resultados de su documento de investigación.
- **Distintas personas:** si el modelo se entrenó con las manos de una sola persona, pruébenlo con otra persona distinta y midan si la precisión cae. Esto es evidencia real de qué tan bien generaliza el sistema.
- **Fondo de la cámara:** MediaPipe es bastante robusto a fondos variados, pero prueben con fondo liso vs. fondo con más objetos/movimiento tras la persona.
- **Métrica a reportar:** usen el *accuracy* y el *classification_report* (precisión, recall, f1-score) que ya imprime `entrenamiento.py` por cada seña — son las métricas estándar para justificar resultados en un proyecto STEM.

---

## 8. Próximos pasos posibles (para la sección de trabajo futuro)

- Agregar reconocimiento de ambas manos (`max_manos=2` en `utils.crear_detector_manos`) para señas bimanuales.
- Incorporar secuencias de frames (en vez de una sola postura) para reconocer señas con movimiento.
- Agregar texto-a-voz (por ejemplo con la librería `pyttsx3`) para que el texto reconocido también se lea en voz alta.
- Integrar este módulo con el dispositivo Arduino de detección de obstáculos en una sola aplicación de "Asistente de discapacidad".
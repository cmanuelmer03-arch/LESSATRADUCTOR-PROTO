¡Bienvenido al proyecto! Este sistema utiliza Inteligencia Artificial 
(Machine Learning y Visión por Computadora) para leer los movimientos 
de tu mano a través de una cámara web y traducirlos a texto.

NOTA: ES UN PROTIPO AVANZADO, PERO REQUIERE MAYOR CANTIDAD DE DATOS Y CONFIGURACIONES
---
Instalar Python (Requisito obligatorio)
Intalar Librerias (información más adelante)
---
------PASOS A SEGUIR-----
-El Entorno Virtual
1. Ve al menú superior y haz clic en **Terminal > Nuevo terminal**.
2. Verás que se abre un panel en la parte inferior. ingresar comando:
python -m venv venv
(Esto crea una carpeta llamada "venv").*

-Activar la Burbuja

En la misma terminal, escribe este comando:
.\venv\Scripts\activate

-ERROR COMÚN EN WINDOWS:
> Si al presionar Enter te sale un error "la ejecución de scripts está deshabilitada"
-En PowerShell con permisos de administrador
-Ejecutar el siguiente comando:  `Set-ExecutionPolicy Unrestricted -Scope CurrentUser`
-Aceptar
-Vuelve a VS Code y repite el comando `.\venv\Scripts\activate`.


Al inicio de la línea de texto en tu terminal ahora aparece la palabra 
**`(venv)`** en color verde. 

---

Instalar las Librerías

Con el `(venv)` verde visible en tu terminal, vamos a descargar los "paquetes de conocimiento" (MediaPipe, OpenCV, etc.). Escribe esto y presiona Enter:
pip install opencv-python mediapipe numpy pandas scikit-learn joblib

Nota: Esto descargará mucha información , puede tardar varios minutos.


---

## FASE 5: Cómo usar el programa (El flujo de trabajo)

¡El sistema ya está listo! Pero recuerda: **El programa viene en blanco**. No sabe ninguna seña, tú tienes que enseñárselas siguiendo estos 3 pasos:

-----Grabar señas----

En la terminal (con el `venv` activo), escribe:
python captura_datos.py

La consola te pedirá un nombre. Escribe por ejemplo: `hola`.
Se abrirá tu cámara. Pon tu mano haciendo la seña de LESSA para "Hola".
Presiona la tecla **`c`** en tu teclado unas 30 veces, moviendo un poquito la mano de lugar y ángulo para que la IA aprenda bien.
Presiona **`n`** para registrar otra seña y repite.
Presiona **`q`** para salir.

----Entrenar el Cerebro ----

Para procesar las imagenes en la terminal escribe:
python entrenamiento.py

----Usar el Traductor (Ejecutar)----

En la terminal escribe:
python main.py

Se abrirá la aplicación principa- 
Dale al botón de encender cámara, haz la seña de comprueba si funcona
la computadora debería de reconocer y la traducir el texto en pantalla.

---

## Solución a problemas comunes

-----La ventana de la cámara no se abre-----

* **Es la cámara incorrecta:** Abre `main.py` y `captura_datos.py`. Busca la línea que dice `cv2.VideoCapture(0)`. Cambia el `0` por un `1` o un `2` y guarda.
* **Permisos de Windows:** Busca en Windows "Configuración de privacidad de la cámara". Asegúrate de tener activada la opción *"Permitir que las aplicaciones de escritorio accedan a la cámara"*.

-----La ventana se quedó trabada y no la puedo cerrar----

Haz clic abajo, dentro de la terminal negra de VS Code, y presiona **`Ctrl + C`**. Esto fuerza a Python a apagar todo de golpe.

------Grabé mal una seña, ¿Cómo la borro?-----

* Para borrar TODO y empezar de cero:** En la carpeta `data/` elimina el archivo `datos_senas.csv`. En la carpeta `modelos/` elimina `modelo_lessa.pkl`.
* Para borrar solo una seña:** Abre `data/datos_senas.csv` en VS Code. Presiona `Ctrl + F`, busca el nombre de tu seña (ej: `gracias`), selecciona todas las filas (líneas) que terminan con esa palabra y bórralas.
* MUY IMPORTANTE:** Si borras algo, **siempre** debes ejecutar `python entrenamiento.py` otra vez para que el cerebro se actualice.

-----¿Qué pasa si paso mi proyecto a otra computadora?----

Si envías tu carpeta completa por correo o USB:

NO debes copiar es la carpeta `venv`.
ERROR COMÚN EN VS CODE ("Import could not be resolved"):**
Si abres un archivo de código y ves que la palabra `mediapipe` está subrayada en amarillo o rojo, significa que VS Code está "mirando" fuera de la burbuja.
> **Solución:**
> 1. Haz clic en el código.
> 2. Presiona en tu teclado `Ctrl + Shift + P`.
> 3. Escribe **`Python: Seleccionar intérprete`** y dale Enter.
> 4. Busca en la lista la opción que diga `Python 3.1x.x ('venv': venv)` y hazle clic. El error desaparecerá.

---
Corrección de Código Obligatoria

Los creadores de la librería de Google (MediaPipe) actualizaron su código recientemente, lo que hace que nuestro programa falle si no le hacemos un pequeño ajuste.

**Hay que corregir dos archivos:**

1. En el panel izquierdo, abre el archivo **`main.py`**.
* Ve a la línea ~42.
* Verás esto: `self.detector_manos = crear_detector_manos(modo_estatico=False)`
* **Cámbialo a:** `self.detector_manos = crear_detector_manos()` (Solo borra lo de adentro de los paréntesis).
* Guarda el archivo (`Ctrl + S`).


2. Ahora abre el archivo **`captura_datos.py`**.
* Ve a la línea ~68.
* Verás esto: `detector = crear_detector_manos(modo_estatico=False)`
* **Cámbialo a:** `detector = crear_detector_manos()`
* Guarda el archivo (`Ctrl + S`).





=====DATOS DE LOS ARCHIVOS=====

captura_datos.py
-----------------
Herramienta para RECOLECTAR datos de entrenamiento propios.

No existe (por ahora) un dataset público de LESSA ya convertido en
"landmarks + etiqueta" listo para entrenar un modelo como este. Lo que
sí existe es el diccionario visual de LESSA (por ejemplo el de la UES).
Este script sirve para que ustedes mismos, siguiendo ese diccionario
como referencia, generen su propio dataset: hacen la seña frente a la
cámara, la etiquetan con su nombre, y el programa guarda el vector de
características correspondiente en un archivo CSV.

Cómo usarlo
-----------
1. Ejecutar el script.
2. Escribir el nombre de la seña que van a capturar (ej: "hola").
3. Colocar la mano frente a la cámara haciendo la seña.
4. Presionar la tecla 'c' para capturar una muestra (repetir 30-50 veces,
   moviendo levemente la mano de posición/ángulo/distancia cada vez, para
   que el modelo aprenda a reconocer la seña de forma robusta).
5. Presionar 'n' para pasar a capturar una seña nueva (pide otro nombre).
6. Presionar 'q' para salir y guardar todo en data/datos_señas.csv.

Recomendaciones para mejores datos:
- Iluminación buena (luz natural, luz artificial, algo de sombra).
- Variar la distancia de la mano a la cámara..

-------------------------------------------------
entrenamiento.py
------------------
Entrena el modelo de clasificación de señas a partir del CSV generado
por captura_datos.py (data/datos_senas.csv) y guarda el modelo entrenado
en modelos/modelo_lessa.pkl.

Se usa un RandomForestClassifier de scikit-learn. Se eligió sobre una
red neuronal profunda / TensorFlow porque:
  - Con pocos datos por clase (decenas o cientos de muestras, no miles)
    un RandomForest generaliza muy bien y es difícil que "sobreajuste".
  - Entrena en segundos en cualquier laptop, sin GPU.
  - Es fácil de interpretar y depurar para un proyecto STEM de bachillerato/
    universidad.
Si en el futuro se recolectan muchos más datos (miles de muestras por
seña) se puede migrar a una red neuronal con TensorFlow/Keras siguiendo
la misma idea: el vector de utils.extraer_caracteristicas() como entrada.

Ejecutar simplemente con:
    python entrenamiento.py

----------------------------------------
main.py
--------
Aplicación de escritorio del Traductor de LESSA.

Muestra el video de la cámara con los puntos de la mano dibujados, corre
el modelo entrenado sobre cada frame y va escribiendo en pantalla el
texto correspondiente a la seña reconocida.

Ejecutar con:
    python main.py

Requiere que ya exista un modelo entrenado en modelos/modelo_lessa.pkl
(ver entrenamiento.py). Si no existe, la app igual abre pero avisa que
falta entrenar el modelo.

----------------------------------------------
"""
modelo.py
----------
Envoltorio (wrapper) alrededor del modelo entrenado. Aísla al resto de
la aplicación (main.py) de los detalles de scikit-learn/joblib: main.py
solo necesita llamar a predecir(vector) y recibir una etiqueta de texto
más un porcentaje de confianza.
"""

"""
utils.py
--------
Funciones auxiliares para el traductor de LESSA.

Aquí vive toda la lógica de "visión por computadora": cómo detectamos la
mano con MediaPipe y cómo convertimos esos 21 puntos (landmarks) en un
vector de números (características) que un modelo de Machine Learning
pueda aprender a clasificar.

NOTA TÉCNICA IMPORTANTE:
Este código usa la API moderna "MediaPipe Tasks" (clase HandLandmarker)
en vez de la antigua "mp.solutions.hands" que aparece en la mayoría de
tutoriales en internet. Google retiró esa API antigua de las versiones
recientes del paquete `mediapipe` de PyPI; si la usan, Python lanza el
error "module 'mediapipe' has no attribute 'solutions'". La forma usada
aquí (HandLandmarker) es la que Google documenta actualmente como
reemplazo oficial.

La primera vez que se ejecuta el programa, esta parte descarga
automáticamente el modelo de detección de manos de Google (un archivo
"hand_landmarker.task" de unos 8 MB) y lo guarda en la carpeta
modelos_mediapipe/. Se necesita internet solo esa primera vez.

No hay que ejecutar este archivo directamente; lo importan los demás
scripts (captura_datos.py, entrenamiento.py, main.py).
"""
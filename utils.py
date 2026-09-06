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

import os
import time
import math
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

# ---------------------------------------------------------------------------
# Modelo de detección de manos (se descarga automáticamente una sola vez)
# ---------------------------------------------------------------------------
_CARPETA_PROYECTO = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO_MEDIAPIPE = os.path.join(_CARPETA_PROYECTO, "modelos_mediapipe", "hand_landmarker.task")
URL_MODELO_MEDIAPIPE = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Conexiones entre los 21 landmarks de una mano (para dibujar el
# "esqueleto" de la mano sobre el video). Es la misma topología que
# usaba la antigua mp.solutions.hands.HAND_CONNECTIONS.
CONEXIONES_MANO = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),           # índice
    (5, 9), (9, 10), (10, 11), (11, 12),      # medio
    (9, 13), (13, 14), (14, 15), (15, 16),    # anular
    (13, 17), (17, 18), (18, 19), (19, 20),   # meñique
    (0, 17),                                   # palma
]

MUÑECA = 0
PUNTAS_DEDOS = {
    "pulgar": 4,
    "indice": 8,
    "medio": 12,
    "anular": 16,
    "meñique": 20,
}

# Tríos de puntos usados para medir el ángulo de flexión de cada dedo
# (articulación_base, articulación_media, punta). Nos dice qué tan
# "doblado" o "estirado" está cada dedo, sin importar en qué parte de la
# imagen esté la mano.
ARTICULACIONES_DEDOS = {
    "pulgar": (2, 3, 4),
    "indice": (5, 6, 8),
    "medio": (9, 10, 12),
    "anular": (13, 14, 16),
    "meñique": (17, 18, 20),
}


def _asegurar_modelo_descargado():
    """Descarga el modelo hand_landmarker.task de Google si todavía no
    existe localmente. Solo necesita ejecutarse una vez por instalación."""
    if os.path.exists(RUTA_MODELO_MEDIAPIPE):
        return
    os.makedirs(os.path.dirname(RUTA_MODELO_MEDIAPIPE), exist_ok=True)
    print("Descargando el modelo de detección de manos de MediaPipe (solo la primera vez, ~8 MB)...")
    urllib.request.urlretrieve(URL_MODELO_MEDIAPIPE, RUTA_MODELO_MEDIAPIPE)
    print("Modelo descargado correctamente.\n")


class DetectorManos:
    """
    Envoltorio sobre HandLandmarker (MediaPipe Tasks) que ofrece una
    interfaz simple para trabajar con video en vivo:

        detector = DetectorManos()
        resultado = detector.procesar(frame_rgb)
        resultado.hand_landmarks   # lista con una entrada por cada mano
                                    # detectada; cada entrada es una lista
                                    # de 21 puntos con atributos .x .y .z
    """

    def __init__(self, max_manos=1, confianza_deteccion=0.6,
                 confianza_presencia=0.5, confianza_seguimiento=0.5):
        _asegurar_modelo_descargado()
        opciones = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=RUTA_MODELO_MEDIAPIPE),
            running_mode=RunningMode.VIDEO,
            num_hands=max_manos,
            min_hand_detection_confidence=confianza_deteccion,
            min_hand_presence_confidence=confianza_presencia,
            min_tracking_confidence=confianza_seguimiento,
        )
        self._detector = HandLandmarker.create_from_options(opciones)
        self._inicio = time.time()
        self._ultimo_timestamp_ms = -1

    def procesar(self, frame_rgb):
        """frame_rgb: imagen en formato RGB (numpy array), tal como la
        entrega cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)."""
        marca_tiempo_ms = int((time.time() - self._inicio) * 1000)
        # detect_for_video exige timestamps estrictamente crecientes.
        if marca_tiempo_ms <= self._ultimo_timestamp_ms:
            marca_tiempo_ms = self._ultimo_timestamp_ms + 1
        self._ultimo_timestamp_ms = marca_tiempo_ms

        imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return self._detector.detect_for_video(imagen_mp, marca_tiempo_ms)

    def cerrar(self):
        self._detector.close()


def crear_detector_manos(max_manos=1, **kwargs):
    """Punto de entrada usado por captura_datos.py y main.py para crear
    el detector de manos."""
    return DetectorManos(max_manos=max_manos, **kwargs)


def dibujar_landmarks(frame, listas_landmarks_manos):
    """Dibuja puntos y líneas de la(s) mano(s) sobre el frame de OpenCV
    (formato BGR). listas_landmarks_manos es resultado.hand_landmarks:
    puede estar vacía (sin manos) o traer una lista de 21 puntos por
    cada mano detectada."""
    if not listas_landmarks_manos:
        return frame
    alto, ancho = frame.shape[:2]
    for landmarks_mano in listas_landmarks_manos:
        puntos_px = [(int(p.x * ancho), int(p.y * alto)) for p in landmarks_mano]
        for a, b in CONEXIONES_MANO:
            cv2.line(frame, puntos_px[a], puntos_px[b], (0, 200, 0), 2)
        for x, y in puntos_px:
            cv2.circle(frame, (x, y), 4, (0, 120, 255), -1)
    return frame


def _distancia(p1, p2):
    """Distancia euclidiana simple entre dos puntos (x, y, z)."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def _angulo(p1, p2, p3):
    """
    Ángulo (en grados) formado en el vértice p2 por los segmentos p2->p1
    y p2->p3. Se usa para medir qué tan flexionado está un dedo: un
    dedo estirado da un ángulo cercano a 180°, uno doblado da un ángulo
    mucho menor.
    """
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    norma = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norma == 0:
        return 0.0
    coseno = np.clip(np.dot(v1, v2) / norma, -1.0, 1.0)  # evita errores de redondeo fuera de [-1, 1]
    return math.degrees(math.acos(coseno))


def landmarks_a_lista(landmarks_mano):
    """landmarks_mano es uno de los elementos de resultado.hand_landmarks:
    una lista de 21 puntos con atributos .x .y .z. Esta función la
    convierte en una lista simple de tuplas (x, y, z)."""
    return [(p.x, p.y, p.z) for p in landmarks_mano]


def extraer_caracteristicas(landmarks_mano):
    """
    Convierte los 21 landmarks de una mano en UN vector de características
    numérico, listo para alimentar al modelo de clasificación.

    El vector combina tres tipos de información, tal como se pidió:
      1. Coordenadas normalizadas de los 21 puntos (posición relativa de
         cada parte de la mano).
      2. Ángulos de flexión de los 5 dedos (¿está estirado o doblado?).
      3. Distancias de cada punta de dedo respecto a la muñeca y entre
         puntas de dedos vecinos (¿están los dedos juntos o separados?).

    Todo se normaliza para que el resultado sea el mismo sin importar si
    la mano está más cerca o más lejos de la cámara, o en qué parte del
    encuadre aparece. Esto es clave: sin esta normalización, el modelo
    "memorizaría" posiciones en pantalla en lugar de aprender la forma
    real de la seña.
    """
    puntos = landmarks_a_lista(landmarks_mano)

    # --- 1) Normalización de coordenadas -----------------------------
    # Trasladamos todo para que la muñeca quede en el origen (0, 0, 0)...
    muñeca = puntos[MUÑECA]
    puntos_centrados = [
        (p[0] - muñeca[0], p[1] - muñeca[1], p[2] - muñeca[2]) for p in puntos
    ]
    # ...y escalamos usando la distancia muñeca -> nudillo del dedo medio
    # como "unidad de referencia", así una mano grande (cerca de cámara)
    # y una pequeña (lejos de cámara) producen el mismo vector si hacen
    # la misma seña.
    referencia = _distancia(puntos_centrados[0], puntos_centrados[9])
    referencia = referencia if referencia > 1e-6 else 1e-6
    puntos_normalizados = [
        (p[0] / referencia, p[1] / referencia, p[2] / referencia) for p in puntos_centrados
    ]
    vector_coordenadas = [coord for p in puntos_normalizados for coord in p]  # 21*3 = 63 valores

    # --- 2) Ángulos de flexión de cada dedo ---------------------------
    vector_angulos = []
    for dedo, (i, j, k) in ARTICULACIONES_DEDOS.items():
        vector_angulos.append(_angulo(puntos[i], puntos[j], puntos[k]))

    # --- 3) Distancias relevantes --------------------------------------
    vector_distancias = []
    # distancia de cada punta de dedo a la muñeca (ya normalizada)
    for dedo, idx in PUNTAS_DEDOS.items():
        vector_distancias.append(_distancia(puntos_normalizados[idx], puntos_normalizados[MUÑECA]))
    # distancia entre puntas de dedos consecutivos (para saber si están
    # separados, como en una "V", o juntos, como en un puño)
    orden = ["pulgar", "indice", "medio", "anular", "meñique"]
    for a, b in zip(orden, orden[1:]):
        vector_distancias.append(
            _distancia(puntos_normalizados[PUNTAS_DEDOS[a]], puntos_normalizados[PUNTAS_DEDOS[b]])
        )

    vector_final = vector_coordenadas + vector_angulos + vector_distancias
    return np.array(vector_final, dtype=np.float32)


def nombres_columnas_caracteristicas():
    """Devuelve los nombres de cada columna del vector de características,
    en el mismo orden que genera extraer_caracteristicas(). Se usa para
    darle encabezados legibles al CSV de datos de entrenamiento."""
    columnas = []
    for i in range(21):
        columnas += [f"p{i}_x", f"p{i}_y", f"p{i}_z"]
    for dedo in ARTICULACIONES_DEDOS:
        columnas.append(f"angulo_{dedo}")
    for dedo in PUNTAS_DEDOS:
        columnas.append(f"dist_{dedo}_muñeca")
    orden = ["pulgar", "indice", "medio", "anular", "meñique"]
    for a, b in zip(orden, orden[1:]):
        columnas.append(f"dist_{a}_{b}")
    return columnas

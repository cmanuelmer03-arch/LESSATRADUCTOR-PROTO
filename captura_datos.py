"""
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
- Varíen la iluminación (luz natural, luz artificial, algo de sombra).
- Varíen la distancia de la mano a la cámara.
- Si varias personas van a usar el sistema, que cada una capture también
  sus propias muestras de cada seña (distintos tamaños de mano ayudan a
  que el modelo generalice mejor).
"""

import os
import csv
import cv2

from utils import crear_detector_manos, dibujar_landmarks, extraer_caracteristicas, nombres_columnas_caracteristicas

RUTA_CSV = os.path.join("data", "datos_senas.csv")


def asegurar_csv():
    """Crea el CSV con encabezados si todavía no existe."""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(RUTA_CSV):
        with open(RUTA_CSV, "w", newline="", encoding="utf-8") as f:
            escritor = csv.writer(f)
            escritor.writerow(["etiqueta"] + nombres_columnas_caracteristicas())


def guardar_muestra(etiqueta, vector_caracteristicas):
    with open(RUTA_CSV, "a", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow([etiqueta] + list(vector_caracteristicas))


def main():
    asegurar_csv()

    etiqueta_actual = input("Nombre de la primera seña a capturar (ej: hola): ").strip().lower()
    contador_muestras = 0

    camara = cv2.VideoCapture(0)
    if not camara.isOpened():
        print("No se pudo abrir la cámara. Verifiquen que no esté siendo usada por otra app.")
        return

    detector = crear_detector_manos()

    print("\nControles:  'c' = capturar muestra  |  'n' = nueva seña  |  'q' = salir\n")

    while True:
        ok, frame = camara.read()
        if not ok:
            print("No se pudo leer un frame de la cámara.")
            break

        frame = cv2.flip(frame, 1)  # efecto espejo, más natural para el usuario
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = detector.procesar(frame_rgb)

        frame = dibujar_landmarks(frame, resultado.hand_landmarks)

        # Overlay de información en pantalla
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Seña actual: {etiqueta_actual}   Muestras: {contador_muestras}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "c=capturar  n=nueva seña  q=salir",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Captura de datos LESSA", frame)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord('c'):
            if resultado.hand_landmarks:
                vector = extraer_caracteristicas(resultado.hand_landmarks[0])
                guardar_muestra(etiqueta_actual, vector)
                contador_muestras += 1
                print(f"  Muestra #{contador_muestras} guardada para '{etiqueta_actual}'")
            else:
                print("  No se detectó ninguna mano, intenten de nuevo.")

        elif tecla == ord('n'):
            nueva = input("\nNombre de la nueva seña a capturar: ").strip().lower()
            if nueva:
                etiqueta_actual = nueva
                contador_muestras = 0

        elif tecla == ord('q'):
            break

    camara.release()
    cv2.destroyAllWindows()
    print(f"\nDatos guardados en {RUTA_CSV}")


if __name__ == "__main__":
    main()

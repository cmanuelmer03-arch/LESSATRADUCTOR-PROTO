"""
Aplicación de escritorio del Traductor de LESSA.
Ejecutar con:
    python main.py
Requiere que ya exista un modelo entrenado en modelos/modelo_lessa.pkl

"""

import tkinter as tk
from collections import deque, Counter

import cv2
from PIL import Image, ImageTk

from utils import crear_detector_manos, dibujar_landmarks, extraer_caracteristicas
from modelo import ClasificadorSenas

# --- Parámetros ajustables de reconocimiento --------------------------------
UMBRAL_CONFIANZA = 0.60          # confianza mínima del modelo para aceptar una predicción
TAMAÑO_BUFFER_ESTABILIZACION = 15  # cuántos frames recientes se consideran para "confirmar" una seña
PROPORCION_MINIMA_ACUERDO = 0.70   # qué % de esos frames debe coincidir para confirmar
FRAMES_SIN_MANO_PARA_REINICIAR = 10  # frames seguidos sin mano antes de permitir repetir la misma seña


class AplicacionLESSA:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("Traductor de Lengua de Señas Salvadoreña (LESSA)")
        self.ventana.geometry("1000x640")
        self.ventana.resizable(False, False)

        self.clasificador = ClasificadorSenas()
        self.detector_manos = crear_detector_manos()
        self.camara = None
        self.camara_activa = False

        self.buffer_predicciones = deque(maxlen=TAMAÑO_BUFFER_ESTABILIZACION)
        self.ultima_seña_confirmada = None
        self.contador_frames_sin_mano = 0

        self._construir_interfaz()

    # ------------------------------------------------------------------
    # Construcción de la interfaz
    # ------------------------------------------------------------------
    def _construir_interfaz(self):
        panel_video = tk.Frame(self.ventana, bg="black", width=640, height=480)
        panel_video.grid(row=0, column=0, rowspan=8, padx=10, pady=10)
        panel_video.grid_propagate(False)

        self.label_video = tk.Label(panel_video, bg="black")
        self.label_video.pack(fill="both", expand=True)

        panel_derecho = tk.Frame(self.ventana)
        panel_derecho.grid(row=0, column=1, sticky="n", padx=10, pady=10)

        tk.Label(panel_derecho, text="Seña detectada", font=("Segoe UI", 12)).pack(anchor="w")
        self.label_seña = tk.Label(panel_derecho, text="—", font=("Segoe UI", 30, "bold"),
                                    fg="#1a73e8", width=13)
        self.label_seña.pack(anchor="w", pady=(0, 5))

        self.label_confianza = tk.Label(panel_derecho, text="Confianza: —", font=("Segoe UI", 10))
        self.label_confianza.pack(anchor="w", pady=(0, 20))

        self.boton_iniciar = tk.Button(panel_derecho, text="Iniciar cámara", width=22,
                                        command=self.iniciar_camara, bg="#34a853", fg="white")
        self.boton_iniciar.pack(pady=3)

        self.boton_detener = tk.Button(panel_derecho, text="Detener cámara", width=22,
                                        command=self.detener_camara, bg="#ea4335", fg="white",
                                        state="disabled")
        self.boton_detener.pack(pady=3)

        tk.Label(panel_derecho, text="Texto traducido", font=("Segoe UI", 12)).pack(
            anchor="w", pady=(25, 5))
        self.caja_texto = tk.Text(panel_derecho, width=30, height=11, wrap="word",
                                   font=("Segoe UI", 11))
        self.caja_texto.pack()

        tk.Button(panel_derecho, text="Limpiar texto", width=22,
                  command=self.limpiar_texto).pack(pady=8)

        self.label_estado = tk.Label(panel_derecho, text="", font=("Segoe UI", 9),
                                      fg="gray", justify="left", wraplength=220)
        self.label_estado.pack(anchor="w", pady=(15, 0))

        if not self.clasificador.disponible():
            self.label_estado.config(
                text="⚠ No se encontró un modelo entrenado todavía.\n"
                     "Ejecuten primero: python captura_datos.py\n"
                     "y luego: python entrenamiento.py",
                fg="#c5221f",
            )


    def iniciar_camara(self):
        if self.camara_activa:
            return
        self.camara = cv2.VideoCapture(0)
        if not self.camara.isOpened():
            self.label_estado.config(text="⚠ No se pudo abrir la cámara.", fg="#c5221f")
            return

        self.camara_activa = True
        self.boton_iniciar.config(state="disabled")
        self.boton_detener.config(state="normal")
        self._actualizar_frame()

    def detener_camara(self):
        self.camara_activa = False
        if self.camara is not None:
            self.camara.release()
            self.camara = None
        self.label_video.configure(image="")
        self.label_seña.config(text="—")
        self.label_confianza.config(text="Confianza: —")
        self.boton_iniciar.config(state="normal")
        self.boton_detener.config(state="disabled")

    def limpiar_texto(self):
        self.caja_texto.delete("1.0", tk.END)


    def _actualizar_frame(self):
        if not self.camara_activa:
            return

        ok, frame = self.camara.read()
        if ok:
            frame = cv2.flip(frame, 1)  # efecto espejo
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultado = self.detector_manos.procesar(frame_rgb)
            frame = dibujar_landmarks(frame, resultado.hand_landmarks)

            if resultado.hand_landmarks:
                self.contador_frames_sin_mano = 0
                if self.clasificador.disponible():
                    vector = extraer_caracteristicas(resultado.hand_landmarks[0])
                    etiqueta, confianza = self.clasificador.predecir(vector)
                    self._procesar_prediccion(etiqueta, confianza)
            else:
                self.contador_frames_sin_mano += 1
                self.buffer_predicciones.append(None)
                self.label_seña.config(text="—")
                self.label_confianza.config(text="Confianza: —")
                if self.contador_frames_sin_mano >= FRAMES_SIN_MANO_PARA_REINICIAR:
                    self.ultima_seña_confirmada = None

            imagen = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imagen_tk = ImageTk.PhotoImage(image=imagen)
            self.label_video.imgtk = imagen_tk  # evita que el garbage collector la borre
            self.label_video.configure(image=imagen_tk)

        self.ventana.after(15, self._actualizar_frame)

    def _procesar_prediccion(self, etiqueta, confianza):

        if confianza < UMBRAL_CONFIANZA:
            self.buffer_predicciones.append(None)
            self.label_seña.config(text=f"{etiqueta} ?")
            self.label_confianza.config(text=f"Confianza: {confianza:.0%} (baja)")
            return

        self.buffer_predicciones.append(etiqueta)
        self.label_seña.config(text=etiqueta)
        self.label_confianza.config(text=f"Confianza: {confianza:.0%}")

        if len(self.buffer_predicciones) == self.buffer_predicciones.maxlen:
            mas_comun, veces = Counter(self.buffer_predicciones).most_common(1)[0]
            proporcion = veces / len(self.buffer_predicciones)

            if (mas_comun is not None
                    and proporcion >= PROPORCION_MINIMA_ACUERDO
                    and mas_comun != self.ultima_seña_confirmada):
                self.caja_texto.insert(tk.END, mas_comun + " ")
                self.caja_texto.see(tk.END)
                self.ultima_seña_confirmada = mas_comun
                self.buffer_predicciones.clear()

    def cerrar(self):
        self.detener_camara()
        self.detector_manos.cerrar()
        self.ventana.destroy()


def main():
    ventana = tk.Tk()
    app = AplicacionLESSA(ventana)
    ventana.protocol("WM_DELETE_WINDOW", app.cerrar)
    ventana.mainloop()


if __name__ == "__main__":
    main()

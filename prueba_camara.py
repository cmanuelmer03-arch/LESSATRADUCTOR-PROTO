import cv2
import mediapipe as mp

print("Librerías importadas correctamente.")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se puede acceder a la cámara")
        break

    cv2.imshow('Prueba de Camara', frame)

    # Presiona la tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
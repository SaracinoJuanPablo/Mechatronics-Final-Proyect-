import socket
import cv2
import numpy as np
import mediapipe as mp

# Configuración de MediaPipe para detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuración del socket UDP
RASPBERRY_IP = '192.168.7.2'  # Dirección IP de tu Raspberry Pi
SEND_PORT = 5001  # Puerto para enviar comandos
RECEIVE_PORT = 5000  # Puerto para recibir video

# Crear sockets UDP
send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receive_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receive_sock.bind(('0.0.0.0', RECEIVE_PORT))

def track_hand():
    # Configuración de MediaPipe Hands más flexible
    with mp_hands.Hands(
        static_image_mode=False,  
        max_num_hands=2,  # Cambiar a 2 para detectar ambas manos
        model_complexity=0,  
        min_detection_confidence=0.5,  # Reducir ligeramente el umbral
        min_tracking_confidence=0.5) as hands:
        
        # Variables para reconstruir el frame
        frame_data = b''
        
        while True:
            # Recibir fragmentos UDP
            data, addr = receive_sock.recvfrom(1500)
            frame_data += data
            
            # Verificar si el frame está completo
            if len(data) < 1400:
                try:
                    # Decodificar frame completo
                    nparr = np.frombuffer(frame_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    # Procesar frame con MediaPipe
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame_rgb)
                    
                    # Bandera para seguimiento
                    right_hand_tracked = False
                    
                    if results.multi_hand_landmarks and results.multi_handedness:
                        # Iterar sobre todas las manos detectadas
                        for idx, hand_handedness in enumerate(results.multi_handedness):
                            # Verificar si es mano derecha
                            if hand_handedness.classification[0].label == 'Left': #como lo detecta espejado usamos la mano izq
                                hand_landmarks = results.multi_hand_landmarks[idx]
                                
                                # Obtener coordenadas de la muñeca
                                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                                
                                # Mapear coordenadas de la palma a un rango de -7.5 a 7.5
                                x_normalized = int((wrist.x - 0.5) * 15)
                                
                                # Enviar comando a la Raspberry Pi
                                send_sock.sendto(str(x_normalized).encode(), (RASPBERRY_IP, SEND_PORT))
                                
                                # Dibujar solo la muñeca y conectarla con una línea simple
                                wrist_pixel = mp_drawing._normalized_to_pixel_coordinates(
                                    wrist.x, wrist.y, frame.shape[1], frame.shape[0]
                                )
                                
                                if wrist_pixel:
                                    cv2.circle(frame, wrist_pixel, 5, (0, 255, 0), -1)
                                    cv2.putText(frame, "Right Hand", (10, 30), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                                (0, 255, 0), 2)
                                
                                right_hand_tracked = True
                                break  # Salir después de procesar la primera mano derecha
                    
                    # Si no se detecta mano derecha, mostrar mensaje
                    if not right_hand_tracked:
                        cv2.putText(frame, "No Right Hand", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                    (0, 0, 255), 2)
                    
                    cv2.imshow('Hand Tracking (Right Hand)', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    # Reiniciar buffer de frame
                    frame_data = b''
                    
                except Exception as e:
                    print(f"Error procesando frame: {e}")
                    frame_data = b''
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        track_hand()
    except KeyboardInterrupt:
        print("Programa detenido")
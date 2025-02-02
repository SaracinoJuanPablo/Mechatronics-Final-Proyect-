import cv2
import numpy as np

def conectar_droidcam(ip_address='192.168.1.34', port='4747'):
    #10.17.103.201 el device ip, wifi ip 192.168.1.34
    """
    Conecta con DroidCam y devuelve el objeto de captura de video.
    
    Parámetros:
    ip_address (str): Dirección IP del teléfono en la red local
    port (str): Puerto de DroidCam (por defecto 4747)
    
    Retorna:
    cv2.VideoCapture: Objeto de captura de video
    """
    # URL del stream de DroidCam
    url = f'http://{ip_address}:{port}/video'
    
    # Crear objeto VideoCapture
    cap = cv2.VideoCapture(url)
    
    if not cap.isOpened():
        raise Exception("No se pudo conectar con DroidCam. Verifica la IP y el puerto.")
    
    return cap

def procesar_video():
    try:
        # Conectar con DroidCam
        # Modifica la IP según la que muestre tu app de DroidCam
        cap = conectar_droidcam(ip_address='192.168.1.34', port='4747')
        
        while True:
            # Leer frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error al obtener el frame")
                break
            
            # Mostrar el frame original
            cv2.imshow('DroidCam', frame)
            
            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error: {str(e)}")
        
    finally:
        # Liberar recursos
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    procesar_video()


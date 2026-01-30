import cv2
import numpy as np
import requests
import time

#--- CONFIGURAÇÕES ---
ESP32_URL = "http://192.168.15.9/predict" # Coloque o IP da sua ESP
CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def resize_com_padding(fatia, size=64):
    """Sua lógica de centralização mantida para garantir a acurácia do treino"""
    h, w = fatia.shape[:2]
    margem_interna = 52 
    escala = margem_interna / max(h, w)
    nova_largura, nova_altura = int(w * escala), int(h * escala)
    
    fatia_redimensionada = cv2.resize(fatia, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)
    fundo_quadrado = np.full((size, size), 255, dtype=np.uint8)
    
    x_offset = (size - nova_largura) // 2
    y_offset = (size - nova_altura) // 2
    fundo_quadrado[y_offset:y_offset+nova_altura, x_offset:x_offset+nova_largura] = fatia_redimensionada
    return fundo_quadrado

def ler_placa_na_esp(caminho_img):
    img_color = cv2.imread(caminho_img)
    if img_color is None: return print("Erro ao carregar imagem.")
    
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidatos = []
    for cnt in contours:
        x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
        # Filtros de tamanho (ajuste conforme necessário)
        if h_c > 30 and w_c > 10: 
            candidatos.append((x_c, y_c, w_c, h_c))
    
    #Ordena da esquerda para a direita (essencial para ler a placa na ordem certa)
    candidatos = sorted(candidatos, key=lambda x: x[0])
    
    placa_final = ""
    print(f"Detectados {len(candidatos)} caracteres. Enviando para ESP32-S3...")

    for i, (x_c, y_c, w_c, h_c) in enumerate(candidatos):
        # Recorte e Preparação
        fatia = img_gray[y_c:y_c+h_c, x_c:x_c+w_c]
        fatia_64 = resize_com_padding(fatia, size=64)
        
        #Converte para bytes brutos (uint8 0-255)
        #A ESP32 fará a normalização (/127.5 - 1.0) internamente no C++
        raw_bytes = fatia_64.tobytes()

        try:
            #Envia para a ESP32
            start_envio = time.time()
            response = requests.post(ESP32_URL, data=raw_bytes, timeout=5)
            
            if response.status_code == 200:
                res = response.json()
                letra = CLASSES[res['index']]
                placa_final += letra
                print(f" Caractere {i+1}: {letra} (Confiança: {res['score']}, IA-Time: {res['time_ms']}ms)")
            else:
                print(f"Erro na ESP32 no caractere {i+1}")
                
        except Exception as e:
            print(f"Falha de conexão: {e}")
            break

    print("\n" + "="*40)
    print(f" :red_car: LEITURA DA PLACA: {placa_final} ")
    print("="*40)

if __name__ == "__main__":
    ler_placa_na_esp("scripts/imgplaca.jpeg")
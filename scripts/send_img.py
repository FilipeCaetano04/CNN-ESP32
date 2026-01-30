import cv2
import numpy as np
import requests
import os

# --- CONFIGURAÇÕES ---
ESP32_URL = "http://172.20.10.2/predict" 
CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
IMG_SIZE = 64
DEBUG_FOLDER = "output_caracteres"

if not os.path.exists(DEBUG_FOLDER):
    os.makedirs(DEBUG_FOLDER)

def resize_com_padding(fatia, size=64):
    h, w = fatia.shape[:2]

    #Baseado no seu treino (96px de letra em 128px de imagem = 0.75)
    #Vamos manter 75% de ocupação para bater com o dataset
    margem_alvo = int(size * 0.75) 

    escala = margem_alvo / max(h, w)
    nova_w, nova_h = int(w * escala), int(h * escala)

    fatia_redimensionada = cv2.resize(fatia, (nova_w, nova_h), interpolation=cv2.INTER_AREA)

    #Binarização agressiva após o resize para evitar anti-aliasing (cinzas)
    _, fatia_redimensionada = cv2.threshold(fatia_redimensionada, 128, 255, cv2.THRESH_BINARY)

    fundo_quadrado = np.full((size, size), 255, dtype=np.uint8)
    x_off = (size - nova_w) // 2
    y_off = (size - nova_h) // 2
    fundo_quadrado[y_off:y_off+nova_h, x_off:x_off+nova_w] = fatia_redimensionada

    return fundo_quadrado


def extrair_caracteres(img_gray):
    # Padroniza largura para estabilizar os filtros
    largura_std = 600
    proporcao = largura_std / float(img_gray.shape[1])
    altura_std = int(img_gray.shape[0] * proporcao)
    img_res = cv2.resize(img_gray, (largura_std, altura_std))

    # Threshold para detecção: O OpenCV prefere achar objetos BRANCOS no fundo PRETO
    # Por isso usamos o THRESH_BINARY_INV aqui apenas para localizar os retângulos
    _, binary_for_contours = cv2.threshold(img_res, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidatos = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(cnt)
        
        if area > 500 and 0.2 < aspect_ratio < 1.0:
            candidatos.append((x, y, w, h))

    candidatos = sorted(candidatos, key=lambda x: x[0])
    return candidatos[:7], img_res

def processar_e_enviar(caminho_img):
    img_color = cv2.imread(caminho_img)
    if img_color is None: return print("Erro ao carregar imagem.")

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    candidatos, img_res = extrair_caracteres(img_gray)
    
    placa_final = ""

    for i, (x, y, w, h) in enumerate(candidatos):
        fatia = img_res[y:y+h, x:x+w]

        #1. Binarização inicial
        _, fatia_bin = cv2.threshold(fatia, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        #2. ENGROSSAR A LETRA (Morfologia)
        #Se a letra na foto estiver muito fina, a IA não reconhece.
        #Usamos um kernel pequeno para dar "peso" ao traço preto.
        kernel = np.ones((2, 2), np.uint8)
        fatia_bin = cv2.erode(fatia_bin, kernel, iterations=1) 

        #3. Redimensionar
        fatia_64 = resize_com_padding(fatia_bin, size=IMG_SIZE)

        #Garante 0 ou 255 absoluto (sem meios-tons)
        fatia_64 = np.where(fatia_64 < 128, 0, 255).astype(np.uint8)

        #DEBUG: Compare este arquivo com suas imagens de treino A e B
        cv2.imwrite(f"{DEBUG_FOLDER}/char{i}.jpg", fatia_64)
        try:
            # Envia os bytes para a ESP32
            response = requests.post(ESP32_URL, data=fatia_64.tobytes(), timeout=5)
            if response.status_code == 200:
                res = response.json()
                letra = CLASSES[res['index']]
                placa_final += letra
                print(f"Detectado [{i+1}/7]: {letra} (Confiança: {res['score']:.2f})")
        except Exception as e:
            print(f"Erro no caractere {i+1}: {e}")

    print(f"\n[ RESULTADO FINAL: {placa_final} ]")

if __name__ == "__main__":
    processar_e_enviar("scripts/imgplaca2.png")
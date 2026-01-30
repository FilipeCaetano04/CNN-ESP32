import requests
import numpy as np
from PIL import Image

# SUBSTITUA PELO IP QUE APARECER NO MONITOR SERIAL DA ESP32
ESP32_URL = "http://192.168.15.9/predict" 

CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def send_image(image_path):
     
#Prepara a imagem (Resize para 64x64 e Grayscale)
    img = Image.open(image_path).convert('L').resize((64, 64))
    raw_bytes = np.array(img, dtype=np.uint8).tobytes()

    print(f"Enviando imagem {image_path} para {ESP32_URL}...")

    try:
         #Envia como binário puro no corpo do POST
        response = requests.post(ESP32_URL, data=raw_bytes, timeout=10)

        if response.status_code == 200:
            res = response.json()
            caractere = CLASSES[res['index']]
            print("-" * 30)
            print(f"✅ RESULTADO: {caractere}")
            print(f"📊 Confiança: {res['score']}")
            print(f"⏱️  Tempo de Inferência: {res['time_ms']} ms")
            print("-" * 30)
        else:
            print(f"❌ Erro no servidor: {response.status_code}")

    except Exception as e:
        print(f"❌ Falha na conexão: {e}")

if __name__ == "__main__":
    send_image('scripts/imgA.jpg')
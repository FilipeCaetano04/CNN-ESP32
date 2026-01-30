#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <inttypes.h>


#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_system.h"

// ===== TFLite Micro =====
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"  // esse é bridge no seu pacote
#include "tensorflow/lite/micro/micro_interpreter.h"                  // sem bridge
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"          // sem bridge
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_log.h"                          // sem bridge






// ===== Seu modelo (TFLite em array .h) =====
#include "modelo_placa_int8.h"

// imagem de teste 64x64 em array .h
#include "img64.h"

// ===== (Opcional) Câmera =====
// OBS: pinout varia por placa/módulo. Deixei o esqueleto.
#define USE_CAMERA 0
#if USE_CAMERA
#include "esp_camera.h"
#endif

static const char* TAG = "placa_tflm";

// ---------- Helpers ----------
static inline int8_t clamp_int8(int v) {
  if (v < -128) return -128;
  if (v > 127) return 127;
  return (int8_t)v;
}

// O seu treino aplica: x = (pixel / 127.5) - 1.0  (pixel em [0..255])
static inline float preprocess_pixel_to_float(uint8_t gray_u8) {
  return (gray_u8 / 127.5f) - 1.0f;  // [-1..1]
}

static inline int8_t quantize_float_to_int8(float x, float scale, int zero_point) {
  // q = round(x/scale) + zp
  int q = (int)lrintf(x / scale) + zero_point;
  return clamp_int8(q);
}

static int tensor_elem_count(const TfLiteTensor* t) {
  int n = 1;
  for (int i = 0; i < t->dims->size; i++) n *= t->dims->data[i];
  return n;
}

// Preenche input com "imagem cinza constante" (teste)
static void fill_input_constant_gray(TfLiteTensor* input, uint8_t gray_u8) {
  // Esperamos input int8 e shape [1,64,64,1]
  const float sc = input->params.scale;
  const int zp = input->params.zero_point;

  const int total = input->bytes;  // bytes == elementos quando int8
  float x = preprocess_pixel_to_float(gray_u8);
  int8_t q = quantize_float_to_int8(x, sc, zp);

  for (int i = 0; i < total; i++) {
    input->data.int8[i] = q;
  }
}

// Preenche input usando imagem 8-bit grayscale (64x64) em array
static void fill_input_from_image(TfLiteTensor* input, const uint8_t* img) {
  const float sc = input->params.scale;
  const int zp = input->params.zero_point;

  const int total = input->bytes; // deve ser 4096

  for (int i = 0; i < total; i++) {
    float x = preprocess_pixel_to_float(img[i]); // 0..255 → [-1..1]
    input->data.int8[i] = quantize_float_to_int8(x, sc, zp);
  }
}

// (Opcional) Converte RGB565 para grayscale 8-bit (aprox)
static inline uint8_t rgb565_to_gray(uint16_t p) {
  // extrai componentes (5/6/5)
  int r = (p >> 11) & 0x1F;
  int g = (p >> 5)  & 0x3F;
  int b = (p)       & 0x1F;
  // expande para 8-bit aproximado
  r = (r * 255) / 31;
  g = (g * 255) / 63;
  b = (b * 255) / 31;
  // luminância simples
  int y = (r * 30 + g * 59 + b * 11) / 100;
  if (y < 0) y = 0;
  if (y > 255) y = 255;
  return (uint8_t)y;
}

#if USE_CAMERA
// TODO: você precisa ajustar este config para o seu módulo S3-CAM específico
static esp_err_t init_camera() {
  camera_config_t config = {};
  // config.ledc_channel = LEDC_CHANNEL_0;
  // config.ledc_timer   = LEDC_TIMER_0;

  // TODO: preencher pinos (D0..D7, PCLK, VSYNC, HREF, XCLK, SDA, SCL, PWDN, RESET)
  // config.pin_d0 = ...; etc.

  // config.xclk_freq_hz = 20000000;
  // config.pixel_format = PIXFORMAT_RGB565; // mais fácil de converter p/ gray
  // config.frame_size   = FRAMESIZE_QVGA;   // depois reduz para 64x64
  // config.fb_count     = 2;
  // config.fb_location  = CAMERA_FB_IN_PSRAM; // recomendado no S3 com PSRAM

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Falha init_camera: %d", err);
  }
  return err;
}
#endif

// ---------- App ----------
extern "C" void app_main(void) {
  ESP_LOGI(TAG, "Boot. Heap free=%lu", (unsigned long)esp_get_free_heap_size());

#if USE_CAMERA
  ESP_ERROR_CHECK(init_camera());
#endif

  // 1) Carregar o modelo do array .h
  const tflite::Model* model = tflite::GetModel(modelo_placa_int8);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG, "Schema incompatível: model=%" PRIu32 " runtime=%" PRIu32, (uint32_t)model->version(), (uint32_t)TFLITE_SCHEMA_VERSION);

    return;
  }

  // 2) Resolver Ops
  // Para começar: AllOpsResolver (mais pesado, mas evita "Op not found")
static tflite::MicroMutableOpResolver<13> resolver;
resolver.AddConv2D();
resolver.AddDepthwiseConv2D();
resolver.AddMaxPool2D();
resolver.AddAveragePool2D();
resolver.AddFullyConnected();
resolver.AddReshape();
resolver.AddSoftmax();
resolver.AddQuantize();
resolver.AddDequantize();
resolver.AddAdd();
resolver.AddMul();
resolver.AddRelu();
resolver.AddMean();

// depois vamos adicionar as ops necessárias aqui

  // 3) Arena na PSRAM (8MB): essencial para CNN
  // Comece com 300KB~800KB dependendo do modelo. Ajuste se AllocateTensors falhar.
  constexpr size_t kTensorArenaSize = 600 * 1024;
  uint8_t* tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!tensor_arena) {
    ESP_LOGE(TAG, "Falha ao alocar tensor_arena (%u bytes) na PSRAM", (unsigned)kTensorArenaSize);
    return;
  }

  // 4) Criar o interpreter (estático para evitar uso de heap interno)
  static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);

  // 5) Alocar tensores
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(TAG, "AllocateTensors falhou. Aumente kTensorArenaSize.");
    return;
  }

  TfLiteTensor* input  = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);

  // Debug: valida shape e quantização
  ESP_LOGI(TAG, "Input: type=%d bytes=%d dims=%d [", input->type, input->bytes, input->dims->size);
  for (int i = 0; i < input->dims->size; i++) {
    printf("%d%s", input->dims->data[i], (i+1<input->dims->size)?",":"");
  }
  printf("] scale=%g zp=%" PRId32 "\n", input->params.scale, (int32_t)input->params.zero_point);

  ESP_LOGI(TAG, "Output: type=%d bytes=%d dims=%d [", output->type, output->bytes, output->dims->size);
  for (int i = 0; i < output->dims->size; i++) {
    printf("%d%s", output->dims->data[i], (i+1<output->dims->size)?",":"");
  }
  printf("] scale=%g zp=%" PRId32 "\n", output->params.scale, (int32_t)output->params.zero_point);

  // Confirma expectativa do treino: 64x64x1
  // (Não vou abortar se diferente; só aviso.)
  if (input->dims->size == 4) {
    int h = input->dims->data[1];
    int w = input->dims->data[2];
    int c = input->dims->data[3];
    if (!(h == 64 && w == 64 && c == 1)) {
      ESP_LOGW(TAG, "Input não é 64x64x1 (é %dx%dx%d). Ajuste preprocess/resize.", h, w, c);
    }
  }

  // Loop de inferência
  while (true) {
#if USE_CAMERA
    // ======== Caminho com câmera (esqueleto) ========
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
      ESP_LOGE(TAG, "Falha ao capturar frame");
      vTaskDelay(pdMS_TO_TICKS(500));
      continue;
    }

    // Se for RGB565: fb->buf contém pixels 16-bit
    // TODO: resize/crop para 64x64 e preencher input->data.int8
    // Exemplo de ideia (NÃO completo):
    // - pegar uma janela central
    // - downsample nearest neighbor
    // - converter para gray_u8
    // - preprocess -> quantize -> escrever no input

    esp_camera_fb_return(fb);

#else
    // ======== Caminho sem câmera (teste com imagem 64x64) ========
    // Modelo lê fundos brancos -> NÃO inverter (invert=false)
    fill_input_from_image(input, g_img64);
#endif

    // Rodar inferência
    if (interpreter.Invoke() != kTfLiteOk) {
      ESP_LOGE(TAG, "Invoke falhou");
      vTaskDelay(pdMS_TO_TICKS(1000));
      continue;
    }

    // Ler output: seu Dense tem 36 classes (softmax), mas a shape pode variar.
    const int n = tensor_elem_count(output);

    int best_idx = 0;
    int8_t best_q = output->data.int8[0];
    for (int i = 1; i < n; i++) {
      int8_t q = output->data.int8[i];
      if (q > best_q) {
        best_q = q;
        best_idx = i;
      }
    }

    // Dequantiza só para facilitar leitura humana (não é “probabilidade” garantida, mas ajuda).
    float best_f = (best_q - output->params.zero_point) * output->params.scale;

    static int loop_count = 0;
    loop_count++;

    if (loop_count % 5 == 0) {
        ESP_LOGI(TAG, "Pred=%d score_q=%d score_f=%g (n=%d)",
                best_idx, (int)best_q, best_f, n);
    }

    vTaskDelay(pdMS_TO_TICKS(5000));
  }
}

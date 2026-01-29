#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <inttypes.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_system.h"

#include "driver/uart.h"

// TFLite Micro
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_log.h"

// Modelo TFLite em .h
#include "modelo_placa_int8.h"

static const char* TAG = "placa_tflm";
const int UART_BUF_SIZE = 1024;
const int IMG_BYTES = 64 * 64;

// Helpers
static inline int8_t clamp_int8(int v) {
  if (v < -128) return -128;
  if (v > 127) return 127;
  return (int8_t)v;
}

// No treino -> [0...255] x = (pixel / 127.5) - 1.0 -> saida [-1...1] 
static inline float preprocess_pixel_to_float(uint8_t gray_u8) {
  return (gray_u8 / 127.5f) - 1.0f; 
}

static inline int8_t quantize_float_to_int8(float x, float scale, int zero_point) {
  // q = round(x/scale) + zp
  int q = (int)lrintf(x / scale) + zero_point;
  return clamp_int8(q);
}

// Função de Inicialização da UART
void init_uart() {
    const uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    
    uart_driver_install(UART_NUM_0, UART_BUF_SIZE * 2, 0, 0, NULL, 0);
    uart_param_config(UART_NUM_0, &uart_config);
}

// App 
extern "C" void app_main(void) {
  
  init_uart();

  ESP_LOGI(TAG, "Boot realizado. Heap free=%lu", (unsigned long)esp_get_free_heap_size());

  // 1) Carregar o modelo do array .h
  const tflite::Model* model = tflite::GetModel(modelo_placa_int8);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG, "Schema incompatível: model=%" PRIu32 " runtime=%" PRIu32, (uint32_t)model->version(), (uint32_t)TFLITE_SCHEMA_VERSION);
    return;
  }

  // 2) Resolver Operações
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

  // 3) Arena na PSRAM (8MB)
  constexpr size_t kTensorArenaSize = 600 * 1024;
  uint8_t* tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  
  if (!tensor_arena) {
    ESP_LOGE(TAG, "Falha ao alocar tensor_arena (%u bytes) na PSRAM", (unsigned)kTensorArenaSize);
    return;
  }

  // 4) Criar o interpreter
  static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);

  // 5) Alocar tensores
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(TAG, "AllocateTensors falhou. Aumente kTensorArenaSize.");
    return;
  }

  TfLiteTensor* input  = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);

  uint8_t rx_buffer[IMG_BYTES];

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

    printf("AGUARDANDO_IMAGEM\n");

    int rx_bytes = uart_read_bytes(UART_NUM_0, rx_buffer, IMG_BYTES, portMAX_DELAY);

    if (rx_bytes == IMG_BYTES) {
        ESP_LOGI(TAG, "Imagem capturada via Serial. Processando...");

        // Passo C: Normalizar e Quantizar cada pixel recebido
        const float sc = input->params.scale;
        const int zp = input->params.zero_point;

        for (int i = 0; i < IMG_BYTES; i++) {
            float x = preprocess_pixel_to_float(rx_buffer[i]);
            input->data.int8[i] = quantize_float_to_int8(x, sc, zp);
        }

        // Passo D: Rodar Inferência
        if (interpreter.Invoke() == kTfLiteOk) {
            int8_t max_score = -128;
            int best_idx = 0;
            for (int i = 0; i < 36; i++) {
                if (output->data.int8[i] > max_score) {
                    max_score = output->data.int8[i];
                    best_idx = i;
                }
            }
            // Passo E: Retornar resultado ao PC
            printf("RESULTADO_PREDICAO:%d:%d\n", best_idx, (int)max_score);
        } else {
            ESP_LOGE(TAG, "Falha no Invoke");
        }
    }

    vTaskDelay(pdMS_TO_TICKS(5000));
  }
}

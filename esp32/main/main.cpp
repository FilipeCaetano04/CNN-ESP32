#include <stdio.h>
#include <string.h>
#include <math.h>
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_http_server.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

// TFLite Micro Headers
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_log.h"

#include "modelo_placa_int8.h"

static const char *TAG = "IA_WIFI_SERVER";
const int IMG_BYTES = 64 * 64; // 4096 bytes

// --- CONFIGURAÇÃO DO SEU WI-FI ---
#define WIFI_SSID      "iPhone (3)"
#define WIFI_PASS      "ldm12345"

// Objetos Globais para o TFLite
tflite::MicroInterpreter* static_interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Helpers de Quantização
static inline int8_t clamp_int8(int v) {
    if (v < -128) return -128;
    if (v > 127) return 127;
    return (int8_t)v;
}

static inline float preprocess_pixel_to_float(uint8_t gray_u8) {
    return (gray_u8 / 127.5f) - 1.0f; 
}

static inline int8_t quantize_float_to_int8(float x, float scale, int zero_point) {
    int q = (int)lrintf(x / scale) + zero_point;
    return clamp_int8(q);
}

// Handler da requisição HTTP POST
esp_err_t predict_handler(httpd_req_t *req) {
    uint8_t *rx_buffer = (uint8_t *)malloc(IMG_BYTES);
    if (!rx_buffer) {
        ESP_LOGE(TAG, "Falha ao alocar buffer de recepção");
        return ESP_FAIL;
    }

    // Recebe os bytes da imagem vindos do Python
    int ret = httpd_req_recv(req, (char *)rx_buffer, IMG_BYTES);
    if (ret <= 0) {
        free(rx_buffer);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "Imagem recebida via Wi-Fi. Processando...");

    // Pré-processamento e Injeção no Tensor de Entrada
    const float sc = input->params.scale;
    const int zp = input->params.zero_point;
    for (int i = 0; i < IMG_BYTES; i++) {
        float normalized = preprocess_pixel_to_float(rx_buffer[i]);
        input->data.int8[i] = quantize_float_to_int8(normalized, sc, zp);
    }
    free(rx_buffer);

    // Medição do Tempo de Inferência
    int64_t start_time = esp_timer_get_time();
    TfLiteStatus invoke_status = static_interpreter->Invoke();
    int64_t end_time = esp_timer_get_time();
    float duration_ms = (end_time - start_time) / 1000.0f;

    if (invoke_status != kTfLiteOk) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    // Encontrar a melhor classe (0-35)
    int8_t max_score = -128;
    int best_idx = 0;
    for (int i = 0; i < 36; i++) {
        if (output->data.int8[i] > max_score) {
            max_score = output->data.int8[i];
            best_idx = i;
        }
    }

    // Enviar resposta JSON
    char resp_str[128];
    snprintf(resp_str, sizeof(resp_str), 
             "{\"index\": %d, \"score\": %d, \"time_ms\": %.2f}", 
             best_idx, (int)max_score, duration_ms);
    
    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, resp_str, strlen(resp_str));
    
    ESP_LOGI(TAG, "Predição enviada: Classe %d, Tempo %.2fms", best_idx, duration_ms);
    return ESP_OK;
}

// Event Handler do Wi-Fi para mostrar o IP no monitor
static void wifi_event_handler(void* arg, esp_event_base_t event_base, int32_t event_id, void* event_data) {
    if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "Conectado! IP: " IPSTR, IP2STR(&event->ip_info.ip));
    }
}

void wifi_init() {
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, NULL));
    
    wifi_config_t wifi_config = {};
    strncpy((char*)wifi_config.sta.ssid, WIFI_SSID, sizeof(wifi_config.sta.ssid));
    strncpy((char*)wifi_config.sta.password, WIFI_PASS, sizeof(wifi_config.sta.password));
    
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_connect());
}

extern "C" void app_main(void) {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    wifi_init();

    // Setup TFLite
    const tflite::Model* model = tflite::GetModel(modelo_placa_int8);
    static tflite::MicroMutableOpResolver<13> resolver;
    resolver.AddConv2D(); resolver.AddDepthwiseConv2D(); resolver.AddMaxPool2D();
    resolver.AddAveragePool2D(); resolver.AddFullyConnected(); resolver.AddReshape();
    resolver.AddSoftmax(); resolver.AddQuantize(); resolver.AddDequantize();
    resolver.AddAdd(); resolver.AddMul(); resolver.AddRelu(); resolver.AddMean();

    size_t arena_size = 600 * 1024;
    uint8_t* arena = (uint8_t*)heap_caps_malloc(arena_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    
    static tflite::MicroInterpreter interpreter(model, resolver, arena, arena_size);
    interpreter.AllocateTensors();
    static_interpreter = &interpreter;
    input = interpreter.input(0);
    output = interpreter.output(0);

    // Iniciar Servidor HTTP na porta 80
    httpd_handle_t server = NULL;
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    if (httpd_start(&server, &config) == ESP_OK) {
        httpd_uri_t uri_predict = {
            .uri = "/predict", .method = HTTP_POST, .handler = predict_handler, .user_ctx = NULL
        };
        httpd_register_uri_handler(server, &uri_predict);
        ESP_LOGI(TAG, "Servidor HTTP iniciado. Endpoint: /predict");
    }
}

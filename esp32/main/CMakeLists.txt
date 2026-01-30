idf_component_register(
    SRCS "main.cpp"
    INCLUDE_DIRS "."
    REQUIRES espressif__esp-tflite-micro esp_driver_uart esp_timer esp_http_server esp_wifi nvs_flash esp_event esp_netif
)

idf_component_register(
    SRCS
        "wifi.cc"
        "main.cc"
    PRIV_INCLUDE_DIRS "."
    PRIV_REQUIRES spi_flash esp_wifi wpa_supplicant nvs_flash
    INCLUDE_DIRS "")

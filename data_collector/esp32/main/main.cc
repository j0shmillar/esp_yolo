#include "wifi.h"

void wifi_task(void) {
    Wifi wifi;

    while (1) {
        vTaskDelay(5000 / portTICK_PERIOD_MS);

        wifi.showRSSI();
    }

    wifi.disconnect();
}

extern "C" void app_main() {
    /* Initialize NVS — it is used to store PHY calibration data */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK( ret );

  xTaskCreate((TaskFunction_t)&wifi_task, "wifi_task", 4 * 1024, NULL, 8, NULL);
  vTaskDelete(NULL);
}

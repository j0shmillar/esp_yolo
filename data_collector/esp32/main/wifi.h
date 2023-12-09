#ifndef __WIFI_H__
#define __WIFI_H__

#include "esp_log.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_wifi.h"
#include "esp_event.h"

#include "nvs_flash.h"

class Wifi {
    public:
        Wifi();
        ~Wifi() = default;

        void connect();
        void disconnect();
        void showRSSI();
};
#endif

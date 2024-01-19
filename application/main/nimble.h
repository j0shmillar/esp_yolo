#ifndef _NIMBLE_H_
#define _NIMBLE_H_
#include "esp_log.h"
#include "nvs_flash.h"
/* BLE */
#include "nimble/ble.h"
#include "esp_peripheral.h"
#include "host/ble_hs.h"
#include "host/ble_uuid.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"
#include "host/ble_hs.h"
#include "host/util/util.h"
#include "console/console.h"
#include "services/gap/ble_svc_gap.h"


#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "esp_nimble_hci.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include <stdbool.h>
#include "nimble/ble.h"
#include "modlog/modlog.h"
// #ifdef __cplusplus
// extern "C" {
// #endif

struct ble_hs_cfg;
struct ble_gatt_register_ctxt;


int gatt_svr_init(void);

void gatt_svr_register_cb(struct ble_gatt_register_ctxt *ctxt, void *arg);
void sendNotification();
void vTasksendNotification(void *pvParameters);
void startBLE();
void stopBLE();
void startNVS();

#endif

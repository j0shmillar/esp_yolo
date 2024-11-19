/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "main_functions.h"

#define QUEUE_LENGTH 1

typedef struct {
  QueueHandle_t xQueue;
} TaskParams_t;

void tf_main(void *pvParameter) 
{
  TaskParams_t *taskParams = (TaskParams_t *)pvParameter;
  QueueHandle_t xQueue = taskParams->xQueue;

  setup();
  while (true) 
  {
    ml_task(xQueue);
  }
}

extern "C" void app_main() {

  gpio_reset_pin((gpio_num_t)4); // GPIO4 controls the 3V3 output
  gpio_set_direction((gpio_num_t)4, GPIO_MODE_OUTPUT);
  gpio_set_level((gpio_num_t)4, 1);

  TaskParams_t taskParams = {
      .xQueue = xQueueCreate(QUEUE_LENGTH, sizeof(int)),
  };

  if (taskParams.xQueue == NULL) {
    ESP_LOGE("app_main", "Error creating the queue");
    return;
  }

  // xTaskCreatePinnedToCore instead of xTaskCreate to split tasks between cores
  // otherwise, core 0 will be overloaded
  xTaskCreatePinnedToCore((TaskFunction_t)&tf_main, "tf_main", 5 * 1024,
                          &taskParams, 8, NULL, 0);

  vTaskDelete(NULL);
}

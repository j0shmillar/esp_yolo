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

#include "esp_camera.h"
#include "esp_log.h"
#include "esp_system.h"
#include "sensor.h"
#include "tensorflow/lite/micro/micro_log.h"

#define CAMERA_FRAME_SIZE FRAMESIZE_96X96
#define CAMERA_PIXEL_FORMAT PIXFORMAT_RGB565 

#define CAMERA_MODULE_NAME "OV2640"
// #define CAMERA_PIN_PWDN  10   // A0?
#define CAMERA_PIN_PWDN  -1
#define CAMERA_PIN_RESET -1
#define CAMERA_PIN_XCLK   2   //XCLK  
#define CAMERA_PIN_SIOD  35   //SDA
#define CAMERA_PIN_SIOC  36   //SCL

#define CAMERA_PIN_D7    45 // (D9)
#define CAMERA_PIN_D6    12 // (D8)
#define CAMERA_PIN_D5    17 // (D7)
#define CAMERA_PIN_D4    18 // (D6)
#define CAMERA_PIN_D3    15 // (D5)
#define CAMERA_PIN_D2    16 // (D4)
#define CAMERA_PIN_D1    37 // (D3)
#define CAMERA_PIN_D0     6 // (D2)
#define CAMERA_PIN_VSYNC  3   //VSYNC
#define CAMERA_PIN_HREF   8   //HS
#define CAMERA_PIN_PCLK   1   //PC

#define XCLK_FREQ_HZ 2000000

#ifdef __cplusplus
extern "C" {
#endif

int app_camera_init();

#ifdef __cplusplus
}
#endif

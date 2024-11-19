# yolov5 for esp32 

this code is adapted from [here](https://github.com/daleonpz/POC_CV_tinyml) - check it out!

## setup
### install the esp-idf

follow the [guide](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/linux-macos-setup.html). 

### data + training
1. generate/download an object detection dataset, in yolo format (you can follow this to obtain a 'person-only' coco dataset)
2. you should now have inside `images/` and `labels/` the dataset split into `train` and `val` (and optional `test`) directories
3. navigate to the YOLOv5 Directory:

```bash
cd yolov5/
```

4. train! your trained model will be saved in under `runs/train/my_run/weights` by default.

```bash
python train.py --img 96 --cfg ../data_collector/model.yaml --batch 32 --epochs 300 --data ../data_collector/model_data.yaml --name my_run
```

5. test!

```bash
python detect.py --source ../data_collector/datasets/images/test/ --weights runs/train/my_run/weights/best.pt --img 96 --name my_run --data ../data_collector/model_data.yaml
```

### quantization and c conversion

1. export model to tflite format and quantize

```bash
python export.py --weights runs/train/my_run/weights/best.pt --include saved_model tflite --img 96 --data ../data_collector/model_data.yaml
```
2. convert quantized model to c array by navigating to `data_collector/datasets/` and running

```bash
python convert_model_to_tflite.py ../../yolov5/runs/train/my_run/weights/best_saved_model/ images/test/<image_name>.jpg
```

## deploy

1. copy `model.cc` to `application/main`
2. update Label Settings - if your labels differ from the default labels, update the `model_settings.cc` file in the `application/main` directory

### Build and Flash the Application

1. navigate `application/` and setup esp-idf

```bash
get_idf
```

3. clean, build, and flash

```bash
idf.py fullclean
idf.py build
idf.py -p flash monitor
```
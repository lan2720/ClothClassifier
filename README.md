# ClothClassifier

## Train
```
python run.py --gpus 0 --save_model_name mobilenetv2 --data_dir [TRAIN_DATA_DIR] --base_model_name mobilenetv2 --mode train
```

## Valid
```
python run.py --gpus 0 --save_model_name mobilenetv2 --data_dir [VALID_DATA_DIR] --base_model_name mobilenetv2 --mode valid
```

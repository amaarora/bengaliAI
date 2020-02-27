export CUDA_VISIBLE_DEVICES=0
export IMAGE_HEIGHT=137
export IMAGE_WIDTH=128
export EPOCHS=10
export TRAIN_BATCH_SIZE=256
export VALIDATION_BATCH_SIZE=256
export IMG_MEAN="(0.485,0.456,0.406)"
export IMG_STD="(0.229,0.224,0.225)"
export BASE_MODEL="resnet34"
export TRAINING_FOLDS="(0,1,2,3)"
export VALIDATION_FOLDS="(4,)"
export TRAINING_FOLDS_CSV="../data/training_folds.csv"

export TRAINING_FOLDS="(0,1,4,3)"
export VALIDATION_FOLDS="(2,)"
python train.py

echo "running new model"
export TRAINING_FOLDS="(0,1,2,4)"
export VALIDATION_FOLDS="(3,)"
python train.py

echo "running new model"
export TRAINING_FOLDS="(0,4,2,3)"
export VALIDATION_FOLDS="(1,)"
python train.py

echo "running new model"
export TRAINING_FOLDS="(4,1,2,3)"
export VALIDATION_FOLDS="(0,)"
python train.py

echo "running new model"
export TRAINING_FOLDS="(0,1,2,3)"
export VALIDATION_FOLDS="(4,)"
python train.py

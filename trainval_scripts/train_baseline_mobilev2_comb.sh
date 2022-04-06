CUDA_VISIBLE_DEVICES=2,3
python train.py models/strongbaseline/mobilenetv2_x1_comb.py\
  --gpus=2,3\
  --workers=16\
  --batch-size=32\
  --exp-name=mobilenetv2\

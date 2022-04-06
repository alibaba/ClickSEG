CUDA_VISIBLE_DEVICES=0,1
python train.py models/strongbaseline/pplcnet_x1_comb_noiter.py\
  --gpus=0,1\
  --workers=16\
  --batch-size=32\
  --exp-name=pplcnet\


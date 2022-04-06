CUDA_VISIBLE_DEVICES=0,1
python train.py models/cdnet/cdnet_res34_sbd.py\
  --gpus=0,1\
  --workers=16\
  --batch-size=32\
  --exp-name=cdnet_res34_sbd\

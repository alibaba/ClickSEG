CUDA_VISIBLE_DEVICES=0,1
python train.py models/focalclick/hrnet32_S2_cclvs.py\
  --gpus=0,1\
  --workers=16\
  --batch-size=32\
  --exp-name=hrnet32_S2_cclvs\

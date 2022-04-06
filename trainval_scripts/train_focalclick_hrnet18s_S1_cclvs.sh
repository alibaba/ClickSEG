CUDA_VISIBLE_DEVICES=0,1
python train.py models/focalclick/hrnet18s_S1_cclvs.py\
  --gpus=0,1\
  --workers=16\
  --batch-size=64\
  --exp-name=hrnet18s_S1_cclvs\

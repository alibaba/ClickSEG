CUDA_VISIBLE_DEVICES=2,3
python train.py models/focalclick/hrnet18s_S2_sbd.py\
  --gpus=2,3\
  --workers=16\
  --batch-size=64\
  --exp-name=hrnet18s_S2_sbd\

CUDA_VISIBLE_DEVICES=0,1
python train.py models/focalclick/segformerB0_S2_sbd.py\
  --gpus=0,1\
  --workers=16\
  --batch-size=64\
  --exp-name=segformerB0_S2_sbd\

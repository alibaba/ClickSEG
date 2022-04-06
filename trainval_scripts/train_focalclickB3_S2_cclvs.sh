CUDA_VISIBLE_DEVICES=2,3
python train.py models/focalclick/segformerB3_S2_cclvs.py\
  --gpus=2,3\
  --workers=16\
  --batch-size=32\
  --exp-name=segformerB3_S2_cclvs\

CUDA_VISIBLE_DEVICES=0,1
python train.py models/focalclick/segformerB3_S2_comb.py\
  --gpus=0,1\
  --workers=16\
  --batch-size=32\
  --exp-name=segformerB3_S2_comb\

CUDA_VISIBLE_DEVICES=0
python train.py models/segformerB0_tubes.py\
  --gpus=0\
  --workers=8\
  --batch-size=24\
  --exp-name=segformerB0_tubes\

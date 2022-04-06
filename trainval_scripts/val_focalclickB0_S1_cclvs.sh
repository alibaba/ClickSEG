python scripts/evaluate_model.py FocalClick\
  --model_dir=./experiments/focalclick/segformerB0_S1_cclvs/000_segformerB0_S1_cclvs/checkpoints/\
  --checkpoint=last_checkpoint\
  --infer-size=256\
  --datasets=GrabCut,Berkeley,DAVIS\
  --gpus=3\
  --n-clicks=20\
  --target-iou=0.90\
  --thresh=0.50\
  #--vis
  #--target-iou=0.95\

#--datasets=GrabCut,Berkeley,DAVIS\ last_checkpoint
#--datasets=DAVIS_high,DAVIS_mid,DAVIS_low\
  


python scripts/evaluate_model.py FocalClick\
  --model_dir=./experiments/focalclick/segformerB3_S2_cclvs/000_segformerB3_S2_cclvs/checkpoints/\
  --checkpoint=last_checkpoint\
  --infer-size=256\
  --datasets=D585_ZERO,D585_SP\
  --gpus=0\
  --n-clicks=20\
  --target-iou=0.95\
  --thresh=0.5\
  #--vis
  #--target-iou=0.95\

#--datasets=GrabCut,Berkeley,PascalVOC,COCO_MVal,SBD,DAVIS,D585_ZERO,D585_SP\
#--datasets=DAVIS_high,DAVIS_mid,DAVIS_low\
  


python scripts/evaluate_model.py FocalClick\
  --model_dir=./experiments/focalclick/hrnet18s_S1_cclvs/005_hrnet18s_S1_cclvs/checkpoints/\
  --checkpoint=last_checkpoint\
  --infer-size=256\
  --datasets=GrabCut,Berkeley,PascalVOC,COCO_MVal,SBD,DAVIS,D585_ZERO,D585_SP\
  --gpus=1\
  --n-clicks=20\
  --target-iou=0.90\
  --thresh=0.5\
  #--vis
  #--target-iou=0.95\

#--datasets=GrabCut,Berkeley,DAVIS\
#--datasets=DAVIS_high,DAVIS_mid,DAVIS_low\
  


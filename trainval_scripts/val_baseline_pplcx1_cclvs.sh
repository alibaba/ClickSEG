python scripts/evaluate_model.py Baseline\
  --model_dir=./experiments/strongbaseline/pplcx1_cclvs/000_pplcnet/checkpoints/\
  --checkpoint=last_checkpoint\
  --infer-size=384\
  --datasets=GrabCut,Berkeley,PascalVOC,COCO_MVal,SBD,DAVIS,D585_ZERO,D585_SP\
  --gpus=1\
  --n-clicks=20\
  --target-iou=0.90\
  --thresh=0.50\
  #--vis
  #--target-iou=0.95\

#--datasets=GrabCut,Berkeley,DAVIS\ last_checkpoint
#--datasets=DAVIS_high,DAVIS_mid,DAVIS_low\
  


python scripts/evaluate_model.py CDNet\
  --model_dir=./experiments/cdnet/cdnet_res34_sbd/000_cdnet_res34_sbd/checkpoints/\
  --checkpoint=last_checkpoint\
  --infer-size=384\
  --datasets=GrabCut,Berkeley,PascalVOC,COCO_MVal,SBD,DAVIS,D585_ZERO,D585_SP\
  --gpus=1\
  --n-clicks=20\
  --target-iou=0.90\
  --thresh=0.5\
  #--vis
  #--target-iou=0.95\

#--datasets=GrabCut,Berkeley,DAVIS\
#--datasets=GrabCut,Berkeley,PascalVOC,COCO_MVal,SBD,DAVIS,D585_ZERO,D585_SP\
  


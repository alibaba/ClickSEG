python scripts/evaluate_model.py FocalClick\
  --checkpoint=/home/admin/workspace/project/codebase/ISeg/OpenSource/Open.v9/weights/focalclick/segformerb0s2/cclvs/last_checkpoint.pth\
  --infer-size=256\
  --target-crop-r=1.4\
  --focus-crop-r=1.4\
  --datasets=DAVIS\
  --gpus=0\
  --n-clicks=20\
  --target-iou=0.90\
  --thresh=0.5\
  --vis
  #--target-iou=0.95\

#--datasets=GrabCut,Berkeley,DAVIS\
#--datasets=DAVIS_high,DAVIS_mid,DAVIS_low\
  


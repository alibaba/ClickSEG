# Guidance for Train/Validation


## Evaluate Trained Models
Find a templet in ./trainval_scripts/val_xxx.sh, for example: ./trainval_scripts/val_focalclickB0_S1_cclvs.sh
```
python scripts/evaluate_model.py FocalClick\
  --model_dir=./experiments/focalclick/segformerB0_S1_cclvs/000_segformerB0_S1_cclvs/checkpoints/\
  --checkpoint=last_checkpoint\
  --infer-size=256\
  --datasets=GrabCut,Berkeley,PascalVOC,COCO_MVal,SBD,DAVIS,D585_ZERO,D585_SP\
  --gpus=3\
  --n-clicks=20\
  --target-iou=0.90\
  --thresh=0.50\
  --vis\
  #--vis_path=/xxxx/xxx/xxx/
```
The args could be explained as follows:
```
FocalClick : the pipeline to inference, your could choose from [FocalClick, CDNet, Baseline] for different models.
--model_dir: the path to your models.
--checkpoint: the name of the model that you want to evalute; if "210,220,230", the 3 models 210.pth,220.pth,230.pth would be evaluate in turn.  
--infer-size: The input size during inference; we choose 256 for FocalClick, 384 for Baseline and CDNet.
--vis: visualize the result or not. if  --vis, the visualised result would be found at ./experiments/vis_val/.
--vis_path: you could set the path to save the visualised result, default='./experiments/vis_val/'
```



## Training with existing protocols
you could find a templet in ./trainval_scripts/train_xxx.sh.




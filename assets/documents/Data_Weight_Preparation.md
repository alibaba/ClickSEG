# Guidance for Preparing Datasets and Pretrained Weights



### 1. Download Datasets
You could consider which dataset you would like to train/val, and download several of them in the following links:


Download SBD, GrabCut, Berkeley, DAVIS, Pascal VOC,  COCO, LVIS following [ritm project page][1] 

DAVIS-585: [CliXEG][2]
 
MSRA10K:  https://mmcheng.net/msra10k/

DUTS-TE+TR: http://saliencydetection.net/duts/

HFlicker: https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4

YoutubeVOS: https://competitions.codalab.org/competitions/20127#participate-get-data

ThinObject: https://github.com/liewjunhao/thin-object-selection



[1]:https://github.com/saic-vul/ritm_interactive_segmentation
[2]:https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing



### 2. Download Imagenet Pretrained Weights
SegFormer : https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia   
HRNet : https://github.com/HRNet/HRNet-Image-Classification

### 3. Set Paths in Config
Do not forget to edit the path to datasets and pretrained weights in ./config.yml

***

## Adding your own dataset
If you want to train/val the model on your own dataset.  
1. Find a templet in ./isegm/data/datasets/ and write your own dataset.
2. Add the dataset in ./isegm/data/datasets/\_init\_.py
3. To val on the dataset: edit the function 'get_dataset' in ./isegm/inference/utils.py; add the dataset name in ./trainval_scripts/val_xxxx.sh.
4. To train on the dataset: add the dataset in the corresponding files in ./models.
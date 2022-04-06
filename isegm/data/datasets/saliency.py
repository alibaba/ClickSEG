from pathlib import Path

import cv2
import numpy as np
import os
from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class SaliencyDataset(ISDataset):
    def __init__(self, dataset_paths,
                 images_dir_name='img', masks_dir_name='gt',
                 init_mask_mode = None, **kwargs):
        super(SaliencyDataset, self).__init__(**kwargs)
        self.name = 'Saliency'
        image_mask_dict = {}

        # ====== MSRA-10k ======
        
        MSRA_root = dataset_paths[0] + '/' #'/home/admin/workspace/project/data/datasets/Saliency/MSRA10K_Imgs_GT/Imgs/'
        file_lst = os.listdir(MSRA_root)
        image_lst = [MSRA_root+i for i in file_lst if '.jpg' in i]
        for i in image_lst:
            mask_path = i.replace('.jpg','.png')
            image_mask_dict[i] = mask_path
    
        
        # ===== DUT-TR ========
        TR_root = dataset_paths[1] + '/' #'/home/admin/workspace/project/data/datasets/Saliency/DUTS-TR/DUTS-TR-Image/'
        file_lst = os.listdir(TR_root)
        image_lst = [TR_root+i for i in file_lst if '.jpg' in i]
        for i in image_lst:
            mask_path = i.replace('.jpg','.png').replace('DUTS-TR-Image','DUTS-TR-Mask')
            image_mask_dict[i] = mask_path
        
        # ===== DUT-TE ========
        TE_root = dataset_paths[2] + '/' #'/home/admin/workspace/project/data/datasets/Saliency/DUTS-TE/DUTS-TE-Image/'
        file_lst = os.listdir(TE_root)
        image_lst = [TE_root+i for i in file_lst if '.jpg' in i]
        for i in image_lst:
            mask_path = i.replace('.jpg','.png').replace('DUTS-TE-Image','DUTS-TE-Mask')
            image_mask_dict[i] = mask_path

        
        self.image_mask_dict = image_mask_dict
        self.dataset_samples = list(self.image_mask_dict.keys() )

    def get_sample(self, index) -> DSample:
        image_path = self.dataset_samples[index]
        mask_path = self.image_mask_dict[image_path]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path).astype(np.int32)
        if len(instances_mask.shape) == 3:
            instances_mask = instances_mask[:,:,0]
        instances_mask = instances_mask > 128
        instances_mask = instances_mask.astype(np.int32)


        return DSample(image, instances_mask, objects_ids=[1], sample_id=index)

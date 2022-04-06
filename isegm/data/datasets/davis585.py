from pathlib import Path

import cv2
import numpy as np
import os
from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class Davis585Dataset(ISDataset):
    def __init__(self, dataset_path,
                 images_dir_name='img', masks_dir_name='gt',
                 init_mask_mode = None, **kwargs):
        super(Davis585Dataset, self).__init__(**kwargs)
        self.name = 'D585'+init_mask_mode
        self.dataset_path = dataset_path + '/'
        self.sample_dict = self.generate_sample_dict(self.dataset_path)
        self.dataset_samples = list( self.sample_dict.keys() )
        self.init_mask_mode = init_mask_mode
    
    def generate_sample_dict(self, dataset_path):
        #sample_dict = {'mask_path': ['image_path','stm_init_path','sp_init_path']}
        sample_dict = {}
        sequence_names = os.listdir(dataset_path)
        for sequence_name in sequence_names:
            sequence_dir = self.dataset_path + sequence_name + '/'
            gt_names = os.listdir(sequence_dir)
            gt_names = [i  for i in gt_names if '.png' in i and 'init' not in i]
            for gt_name in gt_names:
                mask_path = sequence_dir + gt_name
                image_name = gt_name.split('_')[-1].replace('.png','.jpg')
                image_path = sequence_dir + image_name
                stm_init_name = 'init_stm_' + gt_name
                stm_init_path = sequence_dir + stm_init_name
                sp_init_name = 'init_sp_' + gt_name
                sp_init_path = sequence_dir + sp_init_name
                sample_dict[mask_path] = [image_path, stm_init_path, sp_init_path]
        return sample_dict


    def get_sample(self, index) -> DSample:
        mask_path = self.dataset_samples[index]
        image_path = self.sample_dict[mask_path][0]
        stm_init_path = self.sample_dict[mask_path][1]
        sp_init_path = self.sample_dict[mask_path][2]
        #print(mask_path,image_path,stm_init_path,sp_init_path)
        #/home/admin/workspace/project/data/datasets/InterDavis/Selected/bear/init_stm_object1_00009.png


        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        instances_mask = cv2.imread(mask_path)[:,:,0] > 128
        stm_init_mask = cv2.imread(stm_init_path)[:,:,0] > 128
        sp_init_mask = cv2.imread(sp_init_path)[:,:,0] > 128

        if self.init_mask_mode == 'sp':
            init_mask  = sp_init_mask
        elif self.init_mask_mode == 'stm':
            init_mask  = stm_init_mask
        elif self.init_mask_mode == 'zero':
            init_mask = None

        return DSample(image, instances_mask, objects_ids=[1], sample_id=index, init_mask=init_mask)

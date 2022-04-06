from pathlib import Path

import cv2
import numpy as np
import os
from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class COCOMValDataset(ISDataset):
    def __init__(self, dataset_path,
                 images_dir_name='img', masks_dir_name='gt',
                 init_mask_mode = None, **kwargs):
        super(COCOMValDataset, self).__init__(**kwargs)
        self.name = 'COCOMVal'
        image_mask_dict = {}

        mval_root = dataset_path + '/'
        image_dir = mval_root + 'img/'
        gt_dir = mval_root + 'gt/'
        file_lst = os.listdir(image_dir)
        image_lst = [image_dir+i for i in file_lst if '.jpg' in i]
        for i in image_lst:
            mask_path = i.replace('.jpg','.png').replace('/img/','/gt/')
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
